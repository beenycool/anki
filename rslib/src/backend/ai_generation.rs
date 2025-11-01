use std::borrow::ToOwned;
use std::collections::BTreeMap;
use std::convert::TryFrom;

use anki_proto::ai_generation as pb;

use crate::ai_generation::config::{AiConfigStore, AiGenerationConfig, ProviderApiKey};
use crate::ai_generation::providers::fetch_models;
use crate::ai_generation::service::{AiGenerationController, GenerationOutcome};
use crate::ai_generation::{
    FilePayload, GeneratedField, GeneratedNote, GeneratedSource, GenerationConstraints,
    GenerationRequest, InputPayload, ProviderKind, StyleExample,
};
use crate::backend::Backend;
use crate::collection::Collection;
use crate::error::InvalidInputError;
use crate::notetype::NotetypeId;
use crate::prelude::*;
use crate::services::{AiGenerationService, BackendAiGenerationService};
use rusqlite::params;

const STYLE_EXAMPLE_LIMIT: usize = 5;

impl BackendAiGenerationService for Backend {
    fn generate_flashcards(
        &self,
        input: pb::GenerateFlashcardsRequest,
    ) -> Result<pb::GenerateFlashcardsResponse> {
        let (config, request) = self.with_col(|col| {
            let mut config = AiConfigStore::load(col)?;
            config.ensure_defaults();

            let constraints = AiGenerationController::build_constraints(&input);
            let provider = resolve_provider_with_config(&input, &config);
            let payload = payload_from_request(&input)?;
            let model_override = constraints.model_override.clone();

            let style_examples = gather_style_examples(col, &constraints, &config)?;
            let request = GenerationRequest::new(
                provider,
                payload,
                constraints,
                None,
                model_override,
                style_examples,
            );
            Ok((config, request))
        })?;

        let outcome = self
            .runtime_handle()
            .block_on(AiGenerationController::generate(&config, request))?;

        Ok(outcome_to_proto(outcome))
    }

    fn get_ai_config(&self) -> Result<pb::AiGenerationConfig> {
        self.with_col(|col| AiGenerationService::get_ai_config(col))
    }

    fn set_ai_config(&self, input: pb::SetAiConfigRequest) -> Result<()> {
        self.with_col(|col| AiGenerationService::set_ai_config(col, input))
    }

    fn list_models(&self, input: pb::ListModelsRequest) -> Result<pb::ListModelsResponse> {
        let (provider, api_key) = self.with_col(|col| prepare_model_request(col, &input))?;
        let models = self
            .runtime_handle()
            .block_on(fetch_models(&provider, api_key))?;
        Ok(pb::ListModelsResponse { models })
    }
}

impl AiGenerationService for crate::collection::Collection {
    fn generate_flashcards(
        &mut self,
        input: pb::GenerateFlashcardsRequest,
    ) -> Result<pb::GenerateFlashcardsResponse> {
        let mut config = AiConfigStore::load(self)?;
        config.ensure_defaults();

        let constraints = AiGenerationController::build_constraints(&input);
        let provider = resolve_provider_with_config(&input, &config);
        let payload = payload_from_request(&input)?;
        let model_override = constraints.model_override.clone();
        let style_examples = gather_style_examples(self, &constraints, &config)?;
        let request = GenerationRequest::new(
            provider,
            payload,
            constraints,
            None,
            model_override,
            style_examples,
        );

        let outcome = run_generation(&config, request)?;
        Ok(outcome_to_proto(outcome))
    }

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

    fn list_models(&mut self, input: pb::ListModelsRequest) -> Result<pb::ListModelsResponse> {
        let (provider, api_key) = prepare_model_request(self, &input)?;
        let models = run_fetch_models(provider, api_key)?;
        Ok(pb::ListModelsResponse { models })
    }
}

fn resolve_provider_with_config(
    request: &pb::GenerateFlashcardsRequest,
    config: &AiGenerationConfig,
) -> ProviderKind {
    resolve_provider_from_value(request.provider, config)
}

fn resolve_provider_from_value(provider: i32, config: &AiGenerationConfig) -> ProviderKind {
    let provider_enum = pb::Provider::try_from(provider).unwrap_or(pb::Provider::Unspecified);

    match provider_enum {
        pb::Provider::Unspecified => config.provider_selected(),
        other => AiGenerationController::resolve_provider(other),
    }
}

fn resolve_api_key_override(
    provided: &str,
    config: &AiGenerationConfig,
    provider: &ProviderKind,
) -> Option<String> {
    let trimmed = provided.trim();
    if trimmed.is_empty() {
        config.api_key_for(provider).map(ToOwned::to_owned)
    } else {
        Some(trimmed.to_string())
    }
}

fn prepare_model_request(
    col: &mut Collection,
    input: &pb::ListModelsRequest,
) -> Result<(ProviderKind, Option<String>)> {
    let mut config = AiConfigStore::load(col)?;
    config.ensure_defaults();

    let provider = resolve_provider_from_value(input.provider, &config);
    let api_key = resolve_api_key_override(&input.api_key, &config, &provider);

    Ok((provider, api_key))
}

fn run_fetch_models(provider: ProviderKind, api_key: Option<String>) -> Result<Vec<String>> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(fetch_models(&provider, api_key))
    } else {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| AnkiError::InvalidInput {
                source: InvalidInputError {
                    message: format!("failed to initialize async runtime: {err}"),
                    source: None,
                    backtrace: None,
                },
            })?;
        runtime.block_on(fetch_models(&provider, api_key))
    }
}

fn run_generation(
    config: &AiGenerationConfig,
    request: GenerationRequest,
) -> Result<GenerationOutcome> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(AiGenerationController::generate(config, request))
    } else {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| AnkiError::InvalidInput {
                source: InvalidInputError {
                    message: format!("failed to initialize async runtime: {err}"),
                    source: None,
                    backtrace: None,
                },
            })?;
        runtime.block_on(AiGenerationController::generate(config, request))
    }
}

fn gather_style_examples(
    col: &mut Collection,
    constraints: &GenerationConstraints,
    config: &AiGenerationConfig,
) -> Result<Vec<StyleExample>> {
    let Some(notetype_id) = effective_note_type_id(constraints, config) else {
        return Ok(Vec::new());
    };

    let notetype = col.get_notetype(notetype_id)?.or_not_found(notetype_id)?;
    if notetype.fields.is_empty() {
        return Ok(Vec::new());
    }

    let field_names: Vec<String> = notetype
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect();

    let raw_fields: Vec<String> = if let Some(deck_id) = constraints.deck_id {
        let mut stmt = col.storage.db.prepare_cached(
            "select n.flds from notes n \
join cards c on c.nid = n.id \
where n.mid = ? and c.did = ? \
group by n.id \
order by n.mod desc \
limit ?",
        )?;
        let rows = stmt
            .query_map(
                params![
                    notetype_id.0,
                    i64::from(deck_id),
                    STYLE_EXAMPLE_LIMIT as i64
                ],
                |row| row.get(0),
            )?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        rows
    } else {
        let mut stmt = col.storage.db.prepare_cached(
            "select n.flds from notes n \
where n.mid = ? \
order by n.mod desc \
limit ?",
        )?;
        let rows = stmt
            .query_map(params![notetype_id.0, STYLE_EXAMPLE_LIMIT as i64], |row| {
                row.get(0)
            })?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        rows
    };

    let mut examples = Vec::new();
    for record in raw_fields {
        let values = split_note_fields(&record);
        let mut fields = Vec::new();
        for (name, value) in field_names.iter().zip(values.iter()) {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            fields.push(GeneratedField::new(name, trimmed));
        }
        if !fields.is_empty() {
            examples.push(StyleExample { fields });
        }
    }

    Ok(examples)
}

fn effective_note_type_id(
    constraints: &GenerationConstraints,
    config: &AiGenerationConfig,
) -> Option<NotetypeId> {
    if let Some(ntid) = constraints.note_type_id {
        Some(ntid)
    } else if constraints.use_default_note_type {
        config.default_note_type_id
    } else {
        None
    }
}

fn split_note_fields(fields: &str) -> Vec<String> {
    fields
        .split('\u{1f}')
        .map(|value| value.to_string())
        .collect()
}

fn payload_from_request(request: &pb::GenerateFlashcardsRequest) -> Result<InputPayload> {
    use pb::InputType;

    match InputType::try_from(request.input_type).unwrap_or(InputType::Unspecified) {
        InputType::Text => {
            let text = request.text.trim();
            if text.is_empty() {
                invalid_input!("no text provided for generation");
            }
            Ok(InputPayload::Text(text.to_string()))
        }
        InputType::Url => {
            let url = request.url.trim();
            if url.is_empty() {
                invalid_input!("no URL provided for generation");
            }
            Ok(InputPayload::Url(url.to_string()))
        }
        InputType::File => {
            let file = request.file.as_ref().or_invalid("missing file payload")?;
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
        InputType::Unspecified => invalid_input!("no input provided for flashcard generation"),
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
        let entry = map
            .entry(key.provider.as_str().to_string())
            .or_insert_with(|| ProviderApiKey::new(key.provider.clone(), None));

        if key.masked {
            continue;
        }

        key.api_key = key.api_key.and_then(|value| {
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
