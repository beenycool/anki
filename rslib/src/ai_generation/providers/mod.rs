use async_trait::async_trait;
use tracing::error;

use crate::ai_generation::input_processor::ProcessedInput;
use crate::ai_generation::{
    AiResult, GenerationConstraints, GenerationRequest, ProviderKind, ProviderResponse,
    StyleExample,
};

pub mod gemini;
pub mod openai;
pub mod openrouter;
pub mod perplexity;

#[async_trait]
pub trait AiProvider: Send + Sync {
    async fn generate(
        &self,
        request: &GenerationRequest,
        input: &ProcessedInput,
    ) -> AiResult<ProviderResponse>;
}

pub fn provider_factory(
    provider: &ProviderKind,
    api_key: Option<String>,
    model: Option<String>,
) -> AiResult<Box<dyn AiProvider>> {
    match provider {
        ProviderKind::Gemini => Ok(Box::new(gemini::GeminiProvider::new(api_key, model)?)),
        ProviderKind::OpenRouter => Ok(Box::new(openrouter::OpenRouterProvider::new(
            api_key, model,
        )?)),
        ProviderKind::Perplexity => Ok(Box::new(perplexity::PerplexityProvider::new(
            api_key, model,
        )?)),
        ProviderKind::OpenAi => Ok(Box::new(openai::OpenAiProvider::new(api_key, model)?)),
        ProviderKind::Custom(name) => crate::invalid_input!(
            "unsupported provider: {name}. Please choose Gemini, OpenRouter, OpenAI, or Perplexity"
        ),
    }
}

pub(crate) fn build_flashcard_prompt(
    input: &ProcessedInput,
    constraints: &GenerationConstraints,
    style_examples: &[StyleExample],
) -> String {
    let requested_cards = constraints
        .max_cards
        .filter(|cards| *cards > 0)
        .unwrap_or(10);

    let source_line = input
        .source_url
        .as_ref()
        .map(|url| format!("Source URL: {url}\n"))
        .unwrap_or_default();

    let prompt_override = constraints
        .prompt_override
        .as_ref()
        .map(|override_text| format!("Additional instructions:\n{override_text}\n\n"))
        .unwrap_or_default();

    let style_hint = if style_examples.is_empty() {
        String::new()
    } else {
        let examples = style_examples
            .iter()
            .map(|example| {
                example
                    .fields
                    .iter()
                    .map(|field| (field.name.clone(), field.value.clone()))
                    .collect::<std::collections::BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();

        match serde_json::to_string_pretty(&examples) {
            Ok(json) => {
                format!("Match the tone and structure of these existing cards:\n{json}\n\n")
            }
            Err(err) => {
                error!("Failed to serialize style examples for prompt: {}", err);
                String::new()
            }
        }
    };

    format!(
        "You are an expert study coach generating Anki flashcards.\n\
Return a JSON array where each item has these keys: \"front\", \"back\", \"source_excerpt\", \"source_url\".\n\
Create between 3 and {requested_cards} high-quality cards covering the most important ideas.\n\
If information is missing, omit the card. Provide concise phrasing suitable for spaced repetition.\n\
{prompt_override}{style_hint}Content starts below:\n<<<\n{source_line}{content}\n>>>",
        content = input.text
    )
}

pub async fn fetch_models(
    provider: &ProviderKind,
    api_key: Option<String>,
) -> AiResult<Vec<String>> {
    match provider {
        ProviderKind::Gemini => {
            let key = require_api_key("Gemini", api_key)?;
            gemini::list_models(&key).await
        }
        ProviderKind::OpenRouter => {
            let key = require_api_key("OpenRouter", api_key)?;
            openrouter::list_models(&key).await
        }
        ProviderKind::Perplexity => {
            let key = require_api_key("Perplexity", api_key)?;
            perplexity::list_models(&key).await
        }
        ProviderKind::OpenAi => {
            let key = require_api_key("OpenAI", api_key)?;
            openai::OpenAiProvider::list_models(&key).await
        }
        ProviderKind::Custom(name) => {
            crate::invalid_input!("model discovery is not supported for provider: {name}")
        }
    }
}

pub(crate) fn require_api_key(provider_name: &str, api_key: Option<String>) -> AiResult<String> {
    match api_key
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
    {
        Some(value) => Ok(value),
        None => crate::invalid_input!("{provider_name} API key is required"),
    }
}
