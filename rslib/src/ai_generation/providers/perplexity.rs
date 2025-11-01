use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ai_generation::input_processor::ProcessedInput;
use crate::ai_generation::{AiResult, GenerationRequest, ProviderResponse};

use super::{build_flashcard_prompt, require_api_key, AiProvider};

const DEFAULT_MODEL: &str = "sonar-reasoning";
const ENDPOINT: &str = "https://api.perplexity.ai/chat/completions";
const SYSTEM_PROMPT: &str = "You generate concise Anki flashcards and respond with JSON only. Each item must include front, back, source_excerpt, source_url.";

pub struct PerplexityProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl PerplexityProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> AiResult<Self> {
        let api_key = require_api_key("Perplexity", api_key)?;
        let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }
}

#[async_trait::async_trait]
impl AiProvider for PerplexityProvider {
    async fn generate(
        &self,
        request: &GenerationRequest,
        input: &ProcessedInput,
    ) -> AiResult<ProviderResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());

        let user_prompt = build_flashcard_prompt(input, &request.constraints);

        let body = PerplexityRequest {
            model: model.clone(),
            messages: vec![
                PerplexityMessage {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                PerplexityMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            max_tokens: Some(1800),
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
        let parsed: PerplexityResponse = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(err) => crate::invalid_input!(err, "Perplexity response was not valid JSON"),
        };

        if let Some(error) = parsed.error {
            let message = error
                .message
                .unwrap_or_else(|| "Perplexity API returned an error".to_string());
            crate::invalid_input!("Perplexity API error: {message}");
        }

        let raw_output = parsed
            .choices
            .into_iter()
            .find_map(|choice| choice.message.map(|message| message.content))
            .unwrap_or_default();

        if raw_output.trim().is_empty() {
            crate::invalid_input!("Perplexity did not return any content");
        }

        Ok(ProviderResponse {
            raw_output,
            model: Some(model),
            tokens_used: parsed
                .usage
                .and_then(|usage| usage.total_tokens.map(|value| value as u32)),
        })
    }
}

#[derive(Debug, Serialize)]
struct PerplexityRequest {
    model: String,
    messages: Vec<PerplexityMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct PerplexityMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct PerplexityResponse {
    #[serde(default)]
    choices: Vec<PerplexityChoice>,
    #[serde(default)]
    usage: Option<PerplexityUsage>,
    #[serde(default)]
    error: Option<PerplexityError>,
}

#[derive(Debug, Deserialize)]
struct PerplexityChoice {
    message: Option<PerplexityMessageResponse>,
}

#[derive(Debug, Deserialize)]
struct PerplexityMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct PerplexityUsage {
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct PerplexityError {
    message: Option<String>,
}


