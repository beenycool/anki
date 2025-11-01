use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ai_generation::input_processor::ProcessedInput;
use crate::ai_generation::{AiResult, GenerationRequest, ProviderResponse};

use super::{build_flashcard_prompt, require_api_key, AiProvider};

const DEFAULT_MODEL: &str = "anthropic/claude-3.5-sonnet";
const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const MODELS_ENDPOINT: &str = "https://openrouter.ai/api/v1/models";
const SYSTEM_PROMPT: &str = "You generate Anki flashcards. Respond with a JSON array only, where each object has keys front, back, source_excerpt, source_url.";

pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> AiResult<Self> {
        let api_key = require_api_key("OpenRouter", api_key)?;
        let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }
}

pub async fn list_models(api_key: &str) -> AiResult<Vec<String>> {
    let client = Client::new();
    let response = client
        .get(MODELS_ENDPOINT)
        .bearer_auth(api_key)
        .send()
        .await?
        .error_for_status()?;

    let bytes = response.bytes().await?;
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(err) => crate::invalid_input!(err, "OpenRouter models response was not valid JSON"),
    };

    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(|msg| msg.as_str())
            .unwrap_or("OpenRouter API returned an error while listing models");
        crate::invalid_input!("{message}");
    }

    let parsed: OpenRouterModelsResponse = match serde_json::from_value(value) {
        Ok(value) => value,
        Err(err) => crate::invalid_input!(err, "OpenRouter models response was not valid JSON"),
    };

    let mut models: Vec<String> = parsed
        .data
        .into_iter()
        .map(|entry| entry.id)
        .filter(|id| !id.trim().is_empty())
        .collect();

    models.sort();
    models.dedup();
    Ok(models)
}

#[async_trait::async_trait]
impl AiProvider for OpenRouterProvider {
    async fn generate(
        &self,
        request: &GenerationRequest,
        input: &ProcessedInput,
    ) -> AiResult<ProviderResponse> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());

        let user_prompt =
            build_flashcard_prompt(input, &request.constraints, &request.style_examples);

        let body = OpenRouterRequest {
            model: model.clone(),
            messages: vec![
                OpenRouterMessage {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                OpenRouterMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            max_tokens: Some(2048),
            temperature: Some(0.3),
        };

        let response = self
            .client
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let bytes = response.bytes().await?;
        let parsed: OpenRouterResponse = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(err) => crate::invalid_input!(err, "OpenRouter response was not valid JSON"),
        };

        if let Some(error) = parsed.error {
            let message = error
                .message
                .unwrap_or_else(|| "OpenRouter API returned an error".to_string());
            crate::invalid_input!("OpenRouter API error: {message}");
        }

        let raw_output = parsed
            .choices
            .into_iter()
            .find_map(|choice| choice.message.map(|message| message.content))
            .unwrap_or_default();

        if raw_output.trim().is_empty() {
            crate::invalid_input!("OpenRouter did not return any content");
        }

        Ok(ProviderResponse {
            raw_output,
            model: Some(model),
            tokens_used: parsed.usage.and_then(|usage| usage.total_tokens),
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    #[serde(default)]
    choices: Vec<OpenRouterChoice>,
    #[serde(default)]
    usage: Option<OpenRouterUsage>,
    #[serde(default)]
    error: Option<OpenRouterError>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: Option<OpenRouterMessageResponse>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterError {
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    #[serde(default)]
    data: Vec<OpenRouterModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModelEntry {
    id: String,
}
