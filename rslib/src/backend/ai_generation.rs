use std::collections::BTreeMap;

use anki_proto::ai_generation as pb;

use crate::ai_generation::config::{AiConfigStore, AiGenerationConfig, ProviderApiKey};
use crate::ai_generation::service::{AiGenerationController, GenerationOutcome};
use crate::ai_generation::{FilePayload, GenerationRequest, GeneratedField, GeneratedNote, GeneratedSource, InputPayload, ProviderKind};
use crate::backend::Backend;
use crate::error::AnkiError;
use crate::prelude::*;
use crate::services::{AiGenerationService, BackendAiGenerationService};

impl BackendAiGenerationService for Backend {
    fn generate_flashcards(
        &self,
        input: pb::GenerateFlashcardsRequest,
    ) -> Result<pb::GenerateFlashcardsResponse> {
        let (config, request) = self.with_col(|col| {
            let mut config = AiConfigStore::load(col)?;
            config.ensure_defaults();

            let mut constraints = AiGenerationController::build_constraints(&input);
            let provider = resolve_provider_with_config(&input, &config);
            let payload = payload_from_request(&input)?;
            let model_override = constraints.model_override.clone();

            let request = GenerationRequest::new(provider, payload, constraints, None, model_override);
            Ok((config, request))
        })?;

        let outcome = self
            .runtime_handle()
            .block_on(AiGenerationController::generate(&config, request))?;

        Ok(outcome_to_proto(outcome))
    }
}

impl AiGenerationService for crate::collection::Collection {
    fn get_ai_config(&mut self) -> Result<pb::AiGenerationConfig> {
        let mut config = AiConfigStore::load(self)?;
        config.ensure_defaults();
        Ok((&config).into())
    }

    fn set_ai_config(&mut self, input: pb::SetAiConfigRequest) -> Result<()> {
        let mut current = AiConfigStore::load(self)?;
        current.ensure_defaults();

        let incoming = input
            .config
            .as_ref()
            .map(AiGenerationConfig::from)
            .unwrap_or_default();

        if let Some(provider) = incoming.selected_provider {
            current.selected_provider = Some(provider);
        }
        current.default_note_type_id = incoming.default_note_type_id;
        current.preferred_model = incoming.preferred_model;
        current.default_max_cards = incoming.default_max_cards;

        if input.persist_api_keys {
            merge_api_keys(&mut current.api_keys, incoming.api_keys);
        }

        current.ensure_defaults();

        AiConfigStore::save(self, &current)
    }
}

fn resolve_provider_with_config(request: &pb::GenerateFlashcardsRequest, config: &AiGenerationConfig) -> ProviderKind {
    let provider_enum = pb::Provider::from_i32(request.provider).unwrap_or(pb::Provider::ProviderUnspecified);

    match provider_enum {
        pb::Provider::ProviderUnspecified => config.provider_selected(),
        other => AiGenerationController::resolve_provider(other),
    }
}

fn payload_from_request(request: &pb::GenerateFlashcardsRequest) -> Result<InputPayload> {
    use pb::InputType;

    match InputType::from_i32(request.input_type).unwrap_or(InputType::InputTypeUnspecified) {
        InputType::InputTypeText => {
            let text = request.text.trim();
            if text.is_empty() {
                invalid_input!("no text provided for generation");
            }
            Ok(InputPayload::Text(text.to_string()))
        }
        InputType::InputTypeUrl => {
            let url = request.url.trim();
            if url.is_empty() {
                invalid_input!("no URL provided for generation");
            }
            Ok(InputPayload::Url(url.to_string()))
        }
        InputType::InputTypeFile => {
            let file = request
                .file
                .as_ref()
                .ok_or_else(|| AnkiError::invalid_input("missing file payload"))?;
            if file.data.is_empty() {
                invalid_input!("file payload was empty");
            }
            let payload = FilePayload::new(
                if file.filename.is_empty() {
                    "upload".to_string()
                } else {
                    file.filename.clone()
                },
                file.data.clone(),
                if file.mimetype.trim().is_empty() {
                    None
                } else {
                    Some(file.mimetype.clone())
                },
            );
            Ok(InputPayload::File(payload))
        }
        InputType::InputTypeUnspecified => invalid_input!("no input provided for flashcard generation"),
    }
}

fn outcome_to_proto(outcome: GenerationOutcome) -> pb::GenerateFlashcardsResponse {
    pb::GenerateFlashcardsResponse {
        notes: outcome.notes.into_iter().map(note_to_proto).collect(),
        raw_response: outcome.raw_response,
    }
}

fn note_to_proto(note: GeneratedNote) -> pb::GeneratedNote {
    pb::GeneratedNote {
        fields: note.fields.into_iter().map(field_to_proto).collect(),
        source: note.source.map(source_to_proto),
        note_type_id: note.note_type_id.map(i64::from).unwrap_or_default(),
        deck_id: note.deck_id.map(i64::from).unwrap_or_default(),
    }
}

fn field_to_proto(field: GeneratedField) -> pb::GeneratedNoteField {
    pb::GeneratedNoteField {
        name: field.name,
        value: field.value,
    }
}

fn source_to_proto(source: GeneratedSource) -> pb::GeneratedNoteSource {
    pb::GeneratedNoteSource {
        url: source.url.unwrap_or_default(),
        title: source.title.unwrap_or_default(),
        excerpt: source.excerpt.unwrap_or_default(),
    }
}

fn merge_api_keys(existing: &mut Vec<ProviderApiKey>, incoming: Vec<ProviderApiKey>) {
    let mut map: BTreeMap<String, ProviderApiKey> = existing
        .drain(..)
        .map(|key| (key.provider.as_str().to_string(), key))
        .collect();

    for mut key in incoming {
        let entry = map.entry(key.provider.as_str().to_string()).or_insert_with(|| {
            ProviderApiKey::new(key.provider.clone(), None)
        });

        if key.masked {
            continue;
        }

        key.api_key = key
            .api_key
            .and_then(|value| {
                let trimmed = value.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            });
        key.masked = key.api_key.is_some();
        *entry = key;
    }

    *existing = map.into_values().collect();
}

