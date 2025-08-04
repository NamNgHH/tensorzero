//! Cohere API provider implementation.
//!
//! This provider implements the Cohere chat API for TensorZero.
//! Supports both regular and streaming chat completions.
//!
//! # Example
//! ```rust,no_run
//! use gateway::inference::providers::{CohereProvider, Provider};
//! use gateway::model::CredentialLocation;
//!
//! let provider = CohereProvider::new(
//!     "command".to_string(),
//!     Some(CredentialLocation::Env("COHERE_API_KEY".to_string()))
//! );
//! ```

use std::{borrow::Cow, sync::OnceLock, time::Duration};

use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;

use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::{
        types::{
            batch::{
                BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
            },
            ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
            ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
            ProviderInferenceResponse, ProviderInferenceResponseArgs,
            ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, TextChunk, Usage,
        },
        InferenceProvider,
    },
    model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider},
    providers::helpers::{
        check_new_tool_call_name, inject_extra_request_data_and_send,
        inject_extra_request_data_and_send_eventsource,
    },
    tool::{ToolCall, ToolCallChunk, ToolChoice},
};

use super::openai::{
    convert_stream_error, get_chat_url, tensorzero_to_openai_messages, OpenAIFunction,
    OpenAIRequestMessage, OpenAISystemRequestMessage, OpenAITool, OpenAIToolType,
};
// API Constants
lazy_static! {
    static ref COHERE_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.cohere.ai/v1").expect("Failed to parse COHERE_API_BASE")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("COHERE_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "Cohere";
pub const PROVIDER_TYPE: &str = "cohere";

// Enhanced Response Types
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]

pub struct CohereProvider {
    model_name: String,
    #[serde(skip)]
    credentials: CohereCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<CohereCredentials> = OnceLock::new();

impl CohereProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CohereCredentials>,
    ) -> Result<Self, Error> {
        let credentials: CohereCredentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(CohereProvider {
            model_name,
            credentials,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum CohereCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for CohereCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(CohereCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(CohereCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(CohereCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Cohere provider".to_string(),
            })),
        }
    }
}

impl CohereCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            CohereCredentials::Static(api_key) => Ok(api_key),
            CohereCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            CohereCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for CohereProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(CohereRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Cohere request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        let request_url = get_chat_url(&COHERE_API_BASE)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            CohereResponseWithMetadata {
                response,
                latency,
                raw_response,
                raw_request,
                generic_request: request,
            }
            .try_into()
        } else {
            handle_cohere_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(raw_request),
                        raw_response: None,
                    })
                })?,
            )
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(CohereRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Cohere request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = get_chat_url(&COHERE_API_BASE)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let stream = stream_cohere(event_source, start_time).peekable();
        Ok((stream, raw_request))
    }
    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Cohere".to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

fn handle_cohere_error(
    response_code: StatusCode,
    response_body: &str,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
    }
}

pub fn stream_cohere(
    mut event_source: EventSource,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            let mut last_tool_name = None;
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<CohereChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
                                raw_response: None,
                            }.into());
                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            cohere_to_tensorzero_chunk(message.data, d, latency, &mut last_tool_name)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

pub(super) fn prepare_cohere_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_openai_messages(message, PROVIDER_TYPE)?);
    }
    if let Some(system_msg) = tensorzero_to_cohere_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

fn tensorzero_to_cohere_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum CohereResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CohereToolChoice {
    Auto,
    None,
    Any,
}

#[derive(Debug, PartialEq, Serialize)]
struct CohereTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<OpenAITool<'a>> for CohereTool<'a> {
    fn from(tool: OpenAITool<'a>) -> Self {
        CohereTool {
            r#type: tool.r#type,
            function: tool.function,
        }
    }
}

fn prepare_cohere_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> Result<(Option<Vec<CohereTool<'a>>>, Option<CohereToolChoice>), Error> {
    match &request.tool_config {
        None => Ok((None, None)),
        Some(tool_config) => match &tool_config.tool_choice {
            ToolChoice::Specific(tool_name) => {
                let tool = tool_config
                    .tools_available
                    .iter()
                    .find(|t| t.name() == tool_name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ToolNotFound {
                            name: tool_name.clone(),
                        })
                    })?;
                let tools = vec![CohereTool::from(OpenAITool::from(tool))];
                Ok((Some(tools), Some(CohereToolChoice::Any)))
            }
            ToolChoice::Auto | ToolChoice::Required => {
                let tools = tool_config
                    .tools_available
                    .iter()
                    .map(|t| CohereTool::from(OpenAITool::from(t)))
                    .collect();
                let tool_choice = match tool_config.tool_choice {
                    ToolChoice::Auto => CohereToolChoice::Auto,
                    ToolChoice::Required => CohereToolChoice::Any,
                    _ => {
                        return Err(ErrorDetails::InvalidTool {
                            message: "Tool choice must be Auto or Required. This is impossible."
                                .to_string(),
                        }
                        .into())
                    }
                };
                Ok((Some(tools), Some(tool_choice)))
            }
            ToolChoice::None => Ok((None, Some(CohereToolChoice::None))),
        },
    }
}

#[derive(Debug, Serialize)]
struct CohereRequest {
    model_name: String,
    prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}
