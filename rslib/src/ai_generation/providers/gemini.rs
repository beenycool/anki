use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ai_generation::input_processor::ProcessedInput;
use crate::ai_generation::{AiResult, GenerationRequest, ProviderResponse};

use super::{build_flashcard_prompt, require_api_key, AiProvider};

const DEFAULT_MODEL: &str = "gemini-1.5-flash";

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> AiResult<Self> {
        let api_key = require_api_key("Gemini", api_key)?;
        let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }
}

#[async_trait::async_trait]
impl AiProvider for GeminiProvider {
    async fn generate(
        &self,
        request: &GenerationRequest,
        input: &ProcessedInput,
    ) -> AiResult<ProviderResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let prompt = build_flashcard_prompt(input, &request.constraints);

        let body = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart { text: prompt }],
            }],
            generation_config: Some(GeminiGenerationConfig {
                response_mime_type: "application/json".to_string(),
            }),
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        );

        let response = self
            .client
            .post(url)
            .query(&[("key", &self.api_key)])
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let bytes = response.bytes().await?;
        let gemini: GeminiResponse = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(err) => crate::invalid_input!(err, "Gemini response was not valid JSON"),
        };

        if let Some(error) = gemini.error {
            let message = error
                .message
                .unwrap_or_else(|| "Gemini API returned an unspecified error".to_string());
            crate::invalid_input!("Gemini API error: {message}");
        }

        let raw_output = gemini
            .candidates
            .into_iter()
            .find_map(|candidate| candidate.content)
            .and_then(|content| {
                content
                    .parts
                    .into_iter()
                    .find_map(|part| part.text)
            })
            .unwrap_or_default();

        if raw_output.trim().is_empty() {
            crate::invalid_input!("Gemini did not return any content");
        }

        Ok(ProviderResponse {
            raw_output,
            model: Some(model),
            tokens_used: None,
        })
    }
}

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "responseMimeType")]
    response_mime_type: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    error: Option<GeminiError>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContentResponse>,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    #[serde(default)]
    parts: Vec<GeminiPartResponse>,
}

#[derive(Debug, Deserialize)]
struct GeminiPartResponse {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    message: Option<String>,
}


