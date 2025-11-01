use crate::ai_generation::config::AiGenerationConfig;
use crate::ai_generation::flashcard_parser;
use crate::ai_generation::input_processor::InputProcessor;
use crate::ai_generation::providers::provider_factory;
use crate::ai_generation::{
    AiResult, GeneratedNote, GenerationConstraints, GenerationRequest, ProviderKind,
};
use crate::notetype::NotetypeId;

pub struct AiGenerationController;

pub struct GenerationOutcome {
    pub notes: Vec<GeneratedNote>,
    pub raw_response: String,
}

impl AiGenerationController {
    pub async fn generate(
        config: &AiGenerationConfig,
        request: GenerationRequest,
    ) -> AiResult<GenerationOutcome> {
        let request = Self::apply_config_defaults(request, config);

        let provider = provider_factory(
            &request.provider,
            request.api_key.clone(),
            request.model.clone(),
        )?;

        let processed_input = InputProcessor::prepare(&request).await?;
        let provider_response = provider.generate(&request, &processed_input).await?;

        let mut notes = flashcard_parser::parse_raw_output(&provider_response.raw_output)?;
        Self::apply_constraints(&mut notes, &request.constraints, config);

        Ok(GenerationOutcome {
            notes,
            raw_response: provider_response.raw_output,
        })
    }

    pub fn build_constraints(
        req: &anki_proto::ai_generation::GenerateFlashcardsRequest,
    ) -> GenerationConstraints {
        GenerationConstraints {
            max_cards: if req.max_cards > 0 {
                Some(req.max_cards)
            } else {
                None
            },
            note_type_id: if req.note_type_id != 0 {
                Some(NotetypeId::from(req.note_type_id))
            } else {
                None
            },
            use_default_note_type: req.use_default_note_type,
            deck_id: if req.deck_id != 0 {
                Some(req.deck_id.into())
            } else {
                None
            },
            prompt_override: if req.prompt_override.trim().is_empty() {
                None
            } else {
                Some(req.prompt_override.clone())
            },
            model_override: if req.model_override.trim().is_empty() {
                None
            } else {
                Some(req.model_override.clone())
            },
        }
    }

    pub fn resolve_provider(provider: anki_proto::ai_generation::Provider) -> ProviderKind {
        match provider {
            anki_proto::ai_generation::Provider::Gemini => ProviderKind::Gemini,
            anki_proto::ai_generation::Provider::Openrouter => ProviderKind::OpenRouter,
            anki_proto::ai_generation::Provider::Perplexity => ProviderKind::Perplexity,
            anki_proto::ai_generation::Provider::Openai => ProviderKind::OpenAi,
            _ => ProviderKind::Gemini,
        }
    }

    fn apply_config_defaults(
        mut request: GenerationRequest,
        config: &AiGenerationConfig,
    ) -> GenerationRequest {
        if request.api_key.is_none() {
            request.api_key = config.api_key_for(&request.provider).map(ToOwned::to_owned);
        }

        if request.model.is_none() {
            request.model = config.preferred_model.clone();
        }

        if request.constraints.max_cards.is_none() {
            request.constraints.max_cards = config.default_max_cards;
        }

        request
    }

    fn apply_constraints(
        notes: &mut Vec<GeneratedNote>,
        constraints: &GenerationConstraints,
        config: &AiGenerationConfig,
    ) {
        let max_cards = constraints
            .max_cards
            .or(config.default_max_cards)
            .filter(|value| *value > 0)
            .map(|value| value as usize);

        if let Some(limit) = max_cards {
            if notes.len() > limit {
                notes.truncate(limit);
            }
        }

        let note_type_id = Self::resolve_note_type_id(constraints, config);
        let deck_id = constraints.deck_id;

        for note in notes.iter_mut() {
            if note.note_type_id.is_none() {
                note.note_type_id = note_type_id;
            }

            if note.deck_id.is_none() {
                note.deck_id = deck_id;
            }

            if let Some(source) = note.source.as_mut() {
                if source
                    .url
                    .as_ref()
                    .map_or(true, |url| url.trim().is_empty())
                    && source
                        .excerpt
                        .as_ref()
                        .map_or(true, |text| text.trim().is_empty())
                    && source
                        .title
                        .as_ref()
                        .map_or(true, |title| title.trim().is_empty())
                {
                    note.source = None;
                }
            }
        }
    }

    fn resolve_note_type_id(
        constraints: &GenerationConstraints,
        config: &AiGenerationConfig,
    ) -> Option<NotetypeId> {
        if let Some(ntid) = constraints.note_type_id {
            return Some(ntid);
        }

        if constraints.use_default_note_type {
            return config.default_note_type_id;
        }

        None
    }
}
