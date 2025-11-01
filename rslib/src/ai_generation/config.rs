use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use anki_proto::ai_generation as pb;

use crate::ai_generation::{AiResult, ProviderKind};
use crate::collection::Collection;
use crate::notetype::NotetypeId;

const CONFIG_KEY: &str = "aiGenerationConfig";

#[derive(Debug, Clone, Default)]
pub struct ProviderApiKey {
    pub provider: ProviderKind,
    pub api_key: Option<String>,
    pub masked: bool,
}

impl ProviderApiKey {
    pub fn new(provider: ProviderKind, api_key: Option<String>) -> Self {
        Self {
            provider,
            masked: api_key.is_some(),
            api_key,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AiGenerationConfig {
    pub selected_provider: Option<ProviderKind>,
    pub default_note_type_id: Option<NotetypeId>,
    pub api_keys: Vec<ProviderApiKey>,
    pub preferred_model: Option<String>,
    pub default_max_cards: Option<u32>,
}

impl AiGenerationConfig {
    pub fn api_key_for(&self, provider: &ProviderKind) -> Option<&str> {
        self.api_keys
            .iter()
            .find(|key| &key.provider == provider)
            .and_then(|entry| entry.api_key.as_deref())
    }

    pub fn provider_selected(&self) -> ProviderKind {
        self.selected_provider
            .clone()
            .unwrap_or(ProviderKind::Gemini)
    }

    pub fn ensure_defaults(&mut self) {
        let mut map: BTreeMap<String, ProviderApiKey> = self
            .api_keys
            .drain(..)
            .map(|mut key| {
                let name = key.provider.as_str().to_string();
                key.provider = ProviderKind::from_str(&name);
                (name, key)
            })
            .collect();

        for kind in [ProviderKind::Gemini, ProviderKind::OpenRouter, ProviderKind::Perplexity] {
            let key = kind.as_str().to_string();
            map.entry(key).or_insert_with(|| ProviderApiKey {
                provider: kind.clone(),
                api_key: None,
                masked: false,
            });
        }

        self.api_keys = map.into_values().collect();
    }
}

impl From<&AiGenerationConfig> for pb::AiGenerationConfig {
    fn from(config: &AiGenerationConfig) -> Self {
        pb::AiGenerationConfig {
            selected_provider: provider_kind_to_proto(config.provider_selected()) as i32,
            default_note_type_id: config.default_note_type_id.map(Into::into).unwrap_or_default(),
            api_keys: config
                .api_keys
                .iter()
                .map(Into::into)
                .collect(),
            preferred_model: config.preferred_model.clone().unwrap_or_default(),
            default_max_cards: config.default_max_cards.unwrap_or_default(),
        }
    }
}

impl From<AiGenerationConfig> for pb::AiGenerationConfig {
    fn from(config: AiGenerationConfig) -> Self {
        (&config).into()
    }
}

impl From<&pb::AiGenerationConfig> for AiGenerationConfig {
    fn from(config: &pb::AiGenerationConfig) -> Self {
        AiGenerationConfig {
            selected_provider: Some(provider_from_proto(config.provider()))
                .filter(|kind| !matches!(kind, ProviderKind::Custom(_))),
            default_note_type_id: (config.default_note_type_id != 0)
                .then(|| NotetypeId::from(config.default_note_type_id)),
            api_keys: config.api_keys.iter().map(Into::into).collect(),
            preferred_model: if config.preferred_model.trim().is_empty() {
                None
            } else {
                Some(config.preferred_model.clone())
            },
            default_max_cards: (config.default_max_cards > 0)
                .then_some(config.default_max_cards),
        }
    }
}

impl From<pb::AiGenerationConfig> for AiGenerationConfig {
    fn from(config: pb::AiGenerationConfig) -> Self {
        (&config).into()
    }
}

impl From<&ProviderApiKey> for pb::ProviderApiKey {
    fn from(key: &ProviderApiKey) -> Self {
        pb::ProviderApiKey {
            provider: provider_kind_to_proto(key.provider.clone()) as i32,
            api_key: key
                .api_key
                .as_ref()
                .filter(|_| !key.masked)
                .cloned()
                .unwrap_or_default(),
            masked: key.masked || key.api_key.is_some(),
        }
    }
}

impl From<ProviderApiKey> for pb::ProviderApiKey {
    fn from(key: ProviderApiKey) -> Self {
        (&key).into()
    }
}

impl From<&pb::ProviderApiKey> for ProviderApiKey {
    fn from(key: &pb::ProviderApiKey) -> Self {
        let api_key = key
            .api_key
            .trim()
            .to_string();
        let api_key = if api_key.is_empty() {
            None
        } else {
            Some(api_key)
        };

        ProviderApiKey {
            provider: provider_from_proto(pb::Provider::from_i32(key.provider).unwrap_or(pb::Provider::ProviderGemini)),
            masked: key.masked,
            api_key,
        }
    }
}

impl From<pb::ProviderApiKey> for ProviderApiKey {
    fn from(key: pb::ProviderApiKey) -> Self {
        (&key).into()
    }
}

fn provider_kind_to_proto(kind: ProviderKind) -> pb::Provider {
    match kind {
        ProviderKind::Gemini => pb::Provider::ProviderGemini,
        ProviderKind::OpenRouter => pb::Provider::ProviderOpenrouter,
        ProviderKind::Perplexity => pb::Provider::ProviderPerplexity,
        ProviderKind::Custom(_) => pb::Provider::ProviderUnspecified,
    }
}

fn provider_from_proto(provider: pb::Provider) -> ProviderKind {
    match provider {
        pb::Provider::ProviderGemini => ProviderKind::Gemini,
        pb::Provider::ProviderOpenrouter => ProviderKind::OpenRouter,
        pb::Provider::ProviderPerplexity => ProviderKind::Perplexity,
        pb::Provider::ProviderUnspecified => ProviderKind::Gemini,
    }
}

/// Persistence helpers for reading and writing AI configuration to the
/// collection config system. Implemented in a later step.
pub struct AiConfigStore;

impl AiConfigStore {
    pub fn load(col: &Collection) -> AiResult<AiGenerationConfig> {
        let stored: Option<StoredAiGenerationConfig> = col.get_config_optional(CONFIG_KEY);

        let mut config = match stored {
            Some(stored) => stored.into(),
            None => AiGenerationConfig::default(),
        };
        config.ensure_defaults();
        Ok(config)
    }

    pub fn save(col: &mut Collection, config: &AiGenerationConfig) -> AiResult<()> {
        let mut merged = config.clone();

        if let Some(existing) = col.get_config_optional::<StoredAiGenerationConfig>(CONFIG_KEY) {
            let previous: AiGenerationConfig = existing.into();
            let mut map: BTreeMap<String, ProviderApiKey> = previous
                .api_keys
                .into_iter()
                .map(|key| (key.provider.as_str().to_string(), key))
                .collect();

            for entry in merged.api_keys.iter_mut() {
                if entry.masked && entry.api_key.is_none() {
                    if let Some(old) = map.get(entry.provider.as_str()) {
                        entry.api_key = old.api_key.clone();
                        entry.masked = old.masked || old.api_key.is_some();
                    }
                }
            }
        }

        let stored = StoredAiGenerationConfig::from(merged);
        col.set_config(CONFIG_KEY, &stored)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StoredAiGenerationConfig {
    selected_provider: Option<String>,
    default_note_type_id: Option<i64>,
    api_keys: Vec<StoredProviderKey>,
    preferred_model: Option<String>,
    default_max_cards: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StoredProviderKey {
    provider: String,
    api_key: Option<String>,
    masked: bool,
}

impl From<StoredAiGenerationConfig> for AiGenerationConfig {
    fn from(value: StoredAiGenerationConfig) -> Self {
        AiGenerationConfig {
            selected_provider: value
                .selected_provider
                .as_deref()
                .map(ProviderKind::from_str),
            default_note_type_id: value
                .default_note_type_id
                .map(NotetypeId::from),
            api_keys: value
                .api_keys
                .into_iter()
                .map(ProviderApiKey::from)
                .collect(),
            preferred_model: value.preferred_model,
            default_max_cards: value.default_max_cards,
        }
    }
}

impl From<AiGenerationConfig> for StoredAiGenerationConfig {
    fn from(value: AiGenerationConfig) -> Self {
        StoredAiGenerationConfig {
            selected_provider: value
                .selected_provider
                .as_ref()
                .map(ProviderKind::as_str)
                .map(ToString::to_string),
            default_note_type_id: value
                .default_note_type_id
                .map(i64::from),
            api_keys: value
                .api_keys
                .into_iter()
                .map(StoredProviderKey::from)
                .collect(),
            preferred_model: value.preferred_model,
            default_max_cards: value.default_max_cards,
        }
    }
}

impl From<StoredProviderKey> for ProviderApiKey {
    fn from(value: StoredProviderKey) -> Self {
        ProviderApiKey {
            provider: ProviderKind::from_str(value.provider.as_str()),
            api_key: value.api_key,
            masked: value.masked,
        }
    }
}

impl From<ProviderApiKey> for StoredProviderKey {
    fn from(value: ProviderApiKey) -> Self {
        StoredProviderKey {
            provider: value.provider.as_str().to_string(),
            api_key: value.api_key,
            masked: value.masked,
        }
    }
}


