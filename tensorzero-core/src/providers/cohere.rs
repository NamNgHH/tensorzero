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
    model_config: String,
    #[serde(skip)]
    credentials: CohereCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<CohereCredentials> = OnceLock::new();

impl CohereProvider {
    pub fn new(
        model_config: String,
        api_key_location: Option<CohereCredentials>
    ) -> Result<Self, Error> {
        let credentials: CohereCredentials =  build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(CohereProvider {
            model_config,
            credentials,
        })
    }

    pub fn model_config(&self) -> &str {
        &self.model_config
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
            model_config,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(CohereRequest::new(&self.model_config, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Cohere request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = get_chat_url(&self.api_base)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());
        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_config,
            request_body,
            request_builder,
        )
        .await?;
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(CohereResponseWithMetadata {
                response,
                latency,
                raw_response,
                raw_request,
                generic_request: request,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                &raw_request,
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ))
        }
    }

    async fn infer_stream<'a>(
            &'a self,
            ModelProviderRequest {
                request,
                provider_name: _,
                model_config,
            }: ModelProviderRequest<'a>,
            http_client: &'a reqwest::Client,
            dynamic_api_keys: &'a InferenceCredentials,
            model_provider: &'a ModelProvider,
        ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
            let request_body = serde_json::to_value(CohereRequest::new(&self.model_config, request)?)
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
                model_config,
                request_body,
                builder,
            )
            .await?;
            let stream = stream_cohere(event_source, start_time).peekable();
            Ok((stream, raw_request))
        }

}
