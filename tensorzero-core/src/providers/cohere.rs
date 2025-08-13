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
        Url::parse("https://api.cohere.ai/v2").expect("Failed to parse COHERE_API_BASE")
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
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
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
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
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
    for message in &request.messages {
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

// Based on Cohere v2 API documentation
#[derive(Debug, Serialize)]
struct CohereRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<CohereResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CohereTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<CohereToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
}

impl<'a> CohereRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<CohereRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(CohereResponseFormat::JsonObject)
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_cohere_messages(request)?;
        let (tools, tool_choice) = prepare_cohere_tools(request)?;

        Ok(CohereRequest {
            messages,
            model,
            temperature: request.temperature,
            p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            stop_sequences: request.borrow_stop_sequences(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct CohereUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<CohereUsage> for Usage {
    fn from(usage: CohereUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct CohereResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct CohereResponseToolCall {
    id: String,
    function: CohereResponseFunctionCall,
}

impl From<CohereResponseToolCall> for ToolCall {
    fn from(cohere_tool_call: CohereResponseToolCall) -> Self {
        ToolCall {
            id: cohere_tool_call.id,
            name: cohere_tool_call.function.name,
            arguments: cohere_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CohereResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereResponseToolCall>>,
}
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum CohereFinishReason {
    Stop,
    Length,
    ModelLength,
    Error,
    ToolCalls,
    #[serde(other)]
    Unknown,
}

impl From<CohereFinishReason> for FinishReason {
    fn from(reason: CohereFinishReason) -> Self {
        match reason {
            CohereFinishReason::Stop => FinishReason::Stop,
            CohereFinishReason::Length => FinishReason::Length,
            CohereFinishReason::ModelLength => FinishReason::Length,
            CohereFinishReason::Error => FinishReason::Unknown,
            CohereFinishReason::ToolCalls => FinishReason::ToolCall,
            CohereFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct CohereResponseChoice {
    index: u8,
    message: CohereResponseMessage,
    finish_reason: CohereFinishReason,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct CohereResponse {
    choices: Vec<CohereResponseChoice>,
    usage: CohereUsage,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct CohereResponseMeta {
    tokens: CohereUsage,
}

pub struct CohereResponseWithMetadata<'a> {
    response: CohereResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<CohereResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: CohereResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let CohereResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }));
        }
        let usage = response.usage.into();
        let CohereResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(text) = message.content {
            if !text.is_empty() {
                content.push(text.into());
            }
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason: Some(finish_reason.into()),
            },
        ))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CohereFunctionCallChunk {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CohereToolCallChunk {
    id: String,
    function: CohereFunctionCallChunk,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CohereDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereToolCallChunk>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct CohereChatChunkChoice {
    delta: CohereDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<CohereFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct CohereChatChunk {
    choices: Vec<CohereChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CohereUsage>,
}

fn cohere_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: CohereChatChunk,
    latency: Duration,
    last_tool_name: &mut Option<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: Some(raw_message.clone()),
        }
        .into());
    }
    let usage = chunk.usage.map(Into::into);
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(choice_finish_reason.into());
        }
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
                    id: "0".to_string(),
                }));
            }
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: tool_call.id,
                    raw_name: check_new_tool_call_name(tool_call.function.name, last_tool_name),
                    raw_arguments: tool_call.function.arguments,
                }));
            }
        }
    }

    Ok(ProviderInferenceResponseChunk::new(
        content,
        usage,
        raw_message,
        latency,
        finish_reason,
    ))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, RequestMessage, Role};
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[test]
    fn test_cohere_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let cohere_request =
            CohereRequest::new("command-a-03-2025", &request_with_tools).unwrap();

        assert_eq!(cohere_request.model, "command-a-03-2025");
        assert_eq!(cohere_request.messages.len(), 1);
        assert_eq!(cohere_request.temperature, Some(0.5));
        assert_eq!(cohere_request.max_tokens, Some(100));
        assert!(!cohere_request.stream);
        assert_eq!(
            cohere_request.response_format,
            Some(CohereResponseFormat::JsonObject)
        );
        assert!(cohere_request.tools.is_some());
        let tools = cohere_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(cohere_request.tool_choice, Some(CohereToolChoice::Any));
    }

    #[test]
    fn test_try_from_cohere_response() {
        // Test case 1: Valid response with content
        let valid_response = CohereResponse {
            choices: vec![CohereResponseChoice {
                index: 0,
                message: CohereResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: CohereFinishReason::Stop,
            }],
            usage: CohereUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };

        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = CohereRequest {
            messages: vec![],
            model: "command-a-03-2025",
            temperature: Some(0.5),
            p: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(CohereResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(CohereResponseWithMetadata {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );

        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = CohereResponse {
            choices: vec![CohereResponseChoice {
                index: 0,
                message: CohereResponseMessage {
                    content: None,
                    tool_calls: Some(vec![CohereResponseToolCall {
                        id: "call1".to_string(),
                        function: CohereResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: CohereFinishReason::ToolCalls,
            }],
            usage: CohereUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
                total_tokens: 40,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }],
            system: Some("test_system".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = CohereRequest {
            messages: vec![],
            model: "command-a-03-2025",
            temperature: Some(0.5),
            p: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(CohereResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(CohereResponseWithMetadata {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec![ContentBlockOutput::ToolCall(ToolCall {
                id: "call1".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            })]
        );
        assert_eq!(inference_response.usage.input_tokens, 15);
        assert_eq!(inference_response.usage.output_tokens, 25);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(110)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.system, Some("test_system".to_string()));
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]
        );
        // Test case 3: Invalid response with no choices
        let invalid_response_no_choices = CohereResponse {
            choices: vec![],
            usage: CohereUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };

        let request_body = CohereRequest {
            messages: vec![],
            model: "command-a-03-2025",
            temperature: Some(0.5),
            p: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(CohereResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };
        let result = ProviderInferenceResponse::try_from(CohereResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        let details = result.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = CohereResponse {
            choices: vec![
                CohereResponseChoice {
                    index: 0,
                    message: CohereResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: CohereFinishReason::Stop,
                },
                CohereResponseChoice {
                    index: 1,
                    message: CohereResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: CohereFinishReason::Stop,
                },
            ],
            usage: CohereUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        };
        let request_body = CohereRequest {
            messages: vec![],
            model: "command-a-03-2025",
            temperature: Some(0.5),
            p: Some(0.9),
            max_tokens: Some(100),
            seed: Some(69),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(CohereResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };
        let result = ProviderInferenceResponse::try_from(CohereResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        let details = result.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_handle_cohere_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_cohere_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        let details = unauthorized.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test forbidden error
        let forbidden = handle_cohere_error(StatusCode::FORBIDDEN, "Forbidden access");
        let details = forbidden.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test rate limit error
        let rate_limit = handle_cohere_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        let details = rate_limit.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test server error
        let server_error = handle_cohere_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        let details = server_error.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Server error");
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }
    }

    #[test]
    fn test_cohere_api_base() {
        assert_eq!(COHERE_API_BASE.as_str(), "https://api.cohere.ai/v2/");
    }

    #[test]
    fn test_credential_to_cohere_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: CohereCredentials = CohereCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, CohereCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = CohereCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, CohereCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = CohereCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, CohereCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = CohereCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
