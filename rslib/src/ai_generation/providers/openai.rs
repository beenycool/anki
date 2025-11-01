use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ai_generation::input_processor::ProcessedInput;
use crate::ai_generation::{AiResult, GenerationRequest, ProviderResponse};

use super::{build_flashcard_prompt, require_api_key, AiProvider};

const DEFAULT_MODEL: &str = "gpt-4o-mini";
const ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const MODELS_ENDPOINT: &str = "https://api.openai.com/v1/models";
const SYSTEM_PROMPT: &str = "You generate Anki flashcards. Respond with a JSON array only, where each object has keys front, back, source_excerpt, source_url.";

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> AiResult<Self> {
        let api_key = require_api_key("OpenAI", api_key)?;
        let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }

    async fn send_request(&self, body: &OpenAiRequest) -> AiResult<OpenAiResponse> {
        let response = self
            .client
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await?
            .error_for_status()?;

        let bytes = response.bytes().await?;
        let parsed: OpenAiResponse = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(err) => crate::invalid_input!(err, "OpenAI response was not valid JSON"),
        };
        parsed.into_result()
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
            Err(err) => crate::invalid_input!(err, "OpenAI models response was not valid JSON"),
        };

        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(|msg| msg.as_str())
                .unwrap_or("OpenAI API returned an error while listing models");
            crate::invalid_input!("{message}");
        }

        let parsed: OpenAiModelsResponse = match serde_json::from_value(value) {
            Ok(value) => value,
            Err(err) => crate::invalid_input!(err, "OpenAI models response was not valid JSON"),
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
}

#[async_trait::async_trait]
impl AiProvider for OpenAiProvider {
    async fn generate(
        &self,
        request: &GenerationRequest,
        input: &ProcessedInput,
    ) -> AiResult<ProviderResponse> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());

        let user_prompt =
            build_flashcard_prompt(input, &request.constraints, &request.style_examples);

        let body = OpenAiRequest {
            model: model.clone(),
            messages: vec![
                OpenAiMessage {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                OpenAiMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            max_tokens: Some(2048),
            temperature: Some(0.3),
        };

        let parsed = self.send_request(&body).await?;

        let raw_output = parsed
            .choices
            .into_iter()
            .find_map(|choice| choice.message.map(|message| message.content))
            .unwrap_or_default();

        if raw_output.trim().is_empty() {
            crate::invalid_input!("OpenAI did not return any content");
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
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    error: Option<OpenAiError>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: Option<OpenAiMessageResponse>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenAiError {
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelsResponse {
    #[serde(default)]
    data: Vec<OpenAiModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelEntry {
    id: String,
}

impl OpenAiResponse {
    fn into_result(self) -> AiResult<OpenAiResponse> {
        if let Some(error) = self.error {
            let message = error
                .message
                .unwrap_or_else(|| "OpenAI API returned an error".to_string());
            crate::invalid_input!("OpenAI API error: {message}");
        }
        Ok(self)
    }
}
