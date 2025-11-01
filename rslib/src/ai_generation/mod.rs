//! AI-assisted flashcard generation with multiple model providers.

use crate::decks::DeckId;
use crate::error::Result;
use crate::notetype::NotetypeId;

pub mod config;
pub mod flashcard_parser;
pub mod input_processor;
pub mod providers;
pub mod service;

/// Convenient alias for results returned by the AI generation layer.
pub type AiResult<T> = Result<T>;

/// Providers that can be used for flashcard generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Gemini,
    OpenRouter,
    Perplexity,
    /// Catch-all for providers not yet known at compile time.
    Custom(String),
}

impl ProviderKind {
    pub fn as_str(&self) -> &str {
        match self {
            ProviderKind::Gemini => "gemini",
            ProviderKind::OpenRouter => "openrouter",
            ProviderKind::Perplexity => "perplexity",
            ProviderKind::Custom(value) => value.as_str(),
        }
    }

    pub fn from_str(value: &str) -> ProviderKind {
        match value {
            "gemini" => ProviderKind::Gemini,
            "openrouter" => ProviderKind::OpenRouter,
            "perplexity" => ProviderKind::Perplexity,
            other => ProviderKind::Custom(other.to_string()),
        }
    }
}

/// Input data supplied by the user.
#[derive(Debug, Clone)]
pub enum InputPayload {
    Text(String),
    Url(String),
    File(FilePayload),
}

/// Describes a user-supplied file.
#[derive(Debug, Clone)]
pub struct FilePayload {
    pub filename: String,
    pub data: Vec<u8>,
    pub mimetype: Option<String>,
}

impl FilePayload {
    pub fn new(filename: String, data: Vec<u8>, mimetype: Option<String>) -> Self {
        Self {
            filename,
            data,
            mimetype,
        }
    }
}

/// Optional hints that constrain how flashcards are generated.
#[derive(Debug, Clone, Default)]
pub struct GenerationConstraints {
    pub max_cards: Option<u32>,
    pub note_type_id: Option<NotetypeId>,
    pub use_default_note_type: bool,
    pub deck_id: Option<DeckId>,
    pub prompt_override: Option<String>,
    pub model_override: Option<String>,
}

/// Complete request passed to the provider layer.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub provider: ProviderKind,
    pub input: InputPayload,
    pub constraints: GenerationConstraints,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

impl GenerationRequest {
    pub fn new(
        provider: ProviderKind,
        input: InputPayload,
        constraints: GenerationConstraints,
        api_key: Option<String>,
        model: Option<String>,
    ) -> Self {
        Self {
            provider,
            input,
            constraints,
            api_key,
            model,
        }
    }
}

/// Raw response returned by a provider before parsing into notes.
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub raw_output: String,
    pub model: Option<String>,
    pub tokens_used: Option<u32>,
}

/// Represents a single generated field on a note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedField {
    pub name: String,
    pub value: String,
}

impl GeneratedField {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Optional metadata about the originating source.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GeneratedSource {
    pub url: Option<String>,
    pub title: Option<String>,
    pub excerpt: Option<String>,
}

/// Represents an entire note to be added to the collection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GeneratedNote {
    pub fields: Vec<GeneratedField>,
    pub source: Option<GeneratedSource>,
    pub note_type_id: Option<NotetypeId>,
    pub deck_id: Option<DeckId>,
}

impl GeneratedNote {
    pub fn new(fields: Vec<GeneratedField>) -> Self {
        Self {
            fields,
            ..Default::default()
        }
    }
}


