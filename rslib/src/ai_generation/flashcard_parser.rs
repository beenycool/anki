use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::ai_generation::{AiResult, GeneratedField, GeneratedNote, GeneratedSource};

/// Parse the raw output from an AI model into structured notes.
///
/// Providers are instructed to return a JSON array of objects, but we try to be
/// forgiving: code-fenced JSON, JSON objects with a top-level `cards` key, and
/// responses that include additional explanatory text are all handled.
pub fn parse_raw_output(raw: &str) -> AiResult<Vec<GeneratedNote>> {
    let trimmed = raw.trim();

    if trimmed.is_empty() {
        crate::invalid_input!("AI response was empty");
    }

    let mut candidates = Vec::new();
    candidates.push(trimmed.to_string());
    if let Some(block) = extract_code_block(trimmed) {
        candidates.push(block);
    }
    if let Some(array) = extract_json_array(trimmed) {
        candidates.push(array);
    }

    for candidate in candidates {
        if candidate.is_empty() {
            continue;
        }

        if let Some(notes) = try_parse_candidate(&candidate)? {
            if !notes.is_empty() {
                return Ok(notes);
            }
        }
    }

    crate::invalid_input!("AI response did not contain any flashcards");
}

fn try_parse_candidate(candidate: &str) -> AiResult<Option<Vec<GeneratedNote>>> {
    match serde_json::from_str::<Value>(candidate) {
        Ok(value) => {
            let Some(card_values) = coerce_to_array(value) else {
                return Ok(None);
            };

            let mut notes = Vec::new();
            for value in card_values {
                if let Some(note) = parse_card(value) {
                    notes.push(note);
                }
            }

            Ok(Some(notes))
        }
        Err(_) => Ok(None),
    }
}

fn coerce_to_array(value: Value) -> Option<Vec<Value>> {
    match value {
        Value::Array(array) => Some(array),
        Value::Object(mut object) => {
            for key in ["cards", "flashcards", "items"] {
                if let Some(Value::Array(array)) = object.remove(key) {
                    return Some(array);
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_card(value: Value) -> Option<GeneratedNote> {
    match value {
        Value::Object(object) => parse_card_object(object),
        Value::String(text) => {
            // Allow providers to return a single "front :: back" string.
            split_front_back(&text)
        }
        _ => None,
    }
}

fn parse_card_object(object: Map<String, Value>) -> Option<GeneratedNote> {
    let mut front: Option<(String, String)> = None;
    let mut back: Option<(String, String)> = None;
    let mut extra_fields: Vec<(String, String)> = Vec::new();
    let mut source_url: Option<String> = None;
    let mut source_excerpt: Option<String> = None;
    let mut source_title: Option<String> = None;

    for (key, value) in object.iter() {
        let lowercase = key.to_ascii_lowercase();

        match lowercase.as_str() {
            "front" | "question" | "prompt" | "q" => {
                if front.is_none() {
                    if let Some(text) = value_to_string(value) {
                        front = Some((key.clone(), text));
                    }
                }
                continue;
            }
            "back" | "answer" | "response" | "a" => {
                if back.is_none() {
                    if let Some(text) = value_to_string(value) {
                        back = Some((key.clone(), text));
                    }
                }
                continue;
            }
            "source_url" | "sourceurl" | "source_link" | "sourcelink" | "url"
            | "link" | "reference_url" | "referenceurl" => {
                if source_url.is_none() {
                    source_url = value_to_string(value);
                }
                continue;
            }
            "source_excerpt" | "sourceexcerpt" | "excerpt" | "context" | "quote"
            | "source_text" | "sourcetext" => {
                if source_excerpt.is_none() {
                    source_excerpt = value_to_string(value);
                }
                continue;
            }
            "source_title" | "sourcetitle" | "title" | "heading" => {
                if source_title.is_none() {
                    source_title = value_to_string(value);
                }
                continue;
            }
            "source" | "reference" | "citation" => {
                if source_url.is_none() || source_excerpt.is_none() || source_title.is_none() {
                    parse_source_object(value, &mut source_url, &mut source_excerpt, &mut source_title);
                }
                continue;
            }
            "fields" => {
                if let Value::Object(map) = value {
                    for (field_name, field_value) in map {
                        if let Some(text) = value_to_string(field_value) {
                            extra_fields.push((field_name.clone(), text));
                        }
                    }
                }
                continue;
            }
            _ => {}
        }

        if let Some(text) = value_to_string(value) {
            extra_fields.push((key.clone(), text));
        }
    }

    if let Some(source) = object.get("source") {
        parse_source_object(source, &mut source_url, &mut source_excerpt, &mut source_title);
    }

    let front = front.or_else(|| extract_from_extra(&mut extra_fields, &["front", "question"]))?;
    let back = back.or_else(|| extract_from_extra(&mut extra_fields, &["back", "answer"]))?;

    let mut seen = HashSet::new();
    let mut fields = Vec::new();

    push_field(&mut fields, &mut seen, front.0, front.1);
    push_field(&mut fields, &mut seen, back.0, back.1);

    for (name, value) in extra_fields {
        push_field(&mut fields, &mut seen, name, value);
    }

    if fields.is_empty() {
        return None;
    }

    let source = if source_url.is_some() || source_excerpt.is_some() || source_title.is_some() {
        Some(GeneratedSource {
            url: source_url,
            title: source_title,
            excerpt: source_excerpt,
        })
    } else {
        None
    };

    Some(GeneratedNote {
        fields,
        source,
        ..Default::default()
    })
}

fn extract_from_extra(
    extra_fields: &mut Vec<(String, String)>,
    aliases: &[&str],
) -> Option<(String, String)> {
    let mut index = None;
    for (idx, (name, value)) in extra_fields.iter().enumerate() {
        let lowercase = name.to_ascii_lowercase();
        if aliases.iter().any(|alias| lowercase == *alias) {
            if !value.trim().is_empty() {
                index = Some(idx);
                break;
            }
        }
    }

    index.map(|idx| extra_fields.remove(idx))
}

fn parse_source_object(
    value: &Value,
    url: &mut Option<String>,
    excerpt: &mut Option<String>,
    title: &mut Option<String>,
) {
    match value {
        Value::String(text) => {
            let text = text.trim();
            if text.is_empty() {
                return;
            }

            if url.is_none() {
                if let Some((prefix, link)) = split_text_and_url(text) {
                    if !prefix.trim().is_empty() && excerpt.is_none() {
                        *excerpt = Some(prefix.trim().to_string());
                    }
                    *url = Some(link.to_string());
                    return;
                }
            }

            if url.is_none() && looks_like_url(text) {
                *url = Some(text.to_string());
            } else if excerpt.is_none() {
                *excerpt = Some(text.to_string());
            }
        }
        Value::Object(map) => {
            if url.is_none() {
                if let Some(link) = map
                    .get("url")
                    .or_else(|| map.get("link"))
                    .or_else(|| map.get("source_url"))
                    .or_else(|| map.get("reference"))
                {
                    if let Some(text) = value_to_string(link) {
                        *url = Some(text);
                    }
                }
            }

            if excerpt.is_none() {
                if let Some(text) = map
                    .get("excerpt")
                    .or_else(|| map.get("source_excerpt"))
                    .or_else(|| map.get("quote"))
                    .and_then(value_to_string)
                {
                    *excerpt = Some(text);
                }
            }

            if title.is_none() {
                if let Some(text) = map
                    .get("title")
                    .or_else(|| map.get("source_title"))
                    .and_then(value_to_string)
                {
                    *title = Some(text);
                }
            }

            // Allow nested "text" or "description" entries to count as excerpts.
            if excerpt.is_none() {
                if let Some(text) = map
                    .get("text")
                    .or_else(|| map.get("description"))
                    .and_then(value_to_string)
                {
                    *excerpt = Some(text);
                }
            }
        }
        _ => {}
    }
}

fn split_front_back(text: &str) -> Option<GeneratedNote> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let separators = ["::", "=>", " - ", " ~ "];
    let mut parts = None;
    for separator in separators {
        if let Some((front, back)) = trimmed.split_once(separator) {
            parts = Some((front.trim(), back.trim()));
            break;
        }
    }

    let (front, back) = parts?;
    if front.is_empty() || back.is_empty() {
        return None;
    }

    Some(GeneratedNote {
        fields: vec![
            GeneratedField::new("Front", front.to_string()),
            GeneratedField::new("Back", back.to_string()),
        ],
        ..Default::default()
    })
}

fn push_field(fields: &mut Vec<GeneratedField>, seen: &mut HashSet<String>, name: String, value: String) {
    let trimmed_value = value.trim();
    if trimmed_value.is_empty() {
        return;
    }

    let key = name.to_ascii_lowercase();
    if seen.insert(key) {
        fields.push(GeneratedField::new(name, trimmed_value.to_string()));
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Array(array) => {
            if array.is_empty() {
                None
            } else {
                serde_json::to_string(array).ok()
            }
        }
        Value::Object(object) => {
            if object.is_empty() {
                None
            } else {
                serde_json::to_string(object).ok()
            }
        }
    }
}

fn extract_code_block(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_start = &text[start + 3..];
    let after_newline = match after_start.find('\n') {
        Some(idx) => &after_start[idx + 1..],
        None => return None,
    };
    let end = after_newline.find("```")?;
    Some(after_newline[..end].trim().to_string())
}

fn extract_json_array(text: &str) -> Option<String> {
    let mut level = 0isize;
    let mut start_index: Option<usize> = None;
    let mut end_index: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        match ch {
            '[' => {
                if level == 0 {
                    start_index = Some(idx);
                }
                level += 1;
            }
            ']' => {
                if level > 0 {
                    level -= 1;
                    if level == 0 {
                        end_index = Some(idx);
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    match (start_index, end_index) {
        (Some(start), Some(end)) if end >= start => Some(text[start..=end].to_string()),
        _ => None,
    }
}

fn split_text_and_url(text: &str) -> Option<(&str, &str)> {
    for prefix in ["https://", "http://"] {
        if let Some(start) = text.find(prefix) {
            let url = text[start..].trim_end_matches(|c: char| c == '.' || c == ',' || c == ';');
            let prefix_text = text[..start].trim();
            return Some((prefix_text, url.trim()));
        }
    }
    None
}

fn looks_like_url(text: &str) -> bool {
    text.starts_with("http://") || text.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_json_array() {
        let raw = r#"[
            {"front": "What is 2+2?", "back": "4", "source_url": "https://example.com"},
            {"question": "Capital of France", "answer": "Paris", "source_excerpt": "European capitals"}
        ]"#;

        let notes = parse_raw_output(raw).unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].fields[0].name, "front");
        assert_eq!(notes[0].fields[1].name, "back");
        assert_eq!(notes[0].source.as_ref().unwrap().url.as_deref(), Some("https://example.com"));
        assert_eq!(notes[1].fields[0].value, "Capital of France");
        assert!(notes[1].source.as_ref().unwrap().excerpt.as_ref().unwrap().contains("European capitals"));
    }

    #[test]
    fn parses_code_fenced_output() {
        let raw = """
```json
[
  {"front": "Q1", "back": "A1"}
]
```
""";

        let notes = parse_raw_output(raw).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].fields[0].value, "Q1");
    }

    #[test]
    fn skips_invalid_cards() {
        let raw = r#"[
            {"front": "Valid", "back": "Answer"},
            {"front": "", "back": ""}
        ]"#;

        let notes = parse_raw_output(raw).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].fields.len(), 2);
    }

    #[test]
    fn parses_source_object() {
        let raw = r#"{
            "cards": [
                {
                    "front": "Fact",
                    "back": "Detail",
                    "source": {
                        "title": "Article",
                        "url": "https://example.org",
                        "excerpt": "Relevant snippet"
                    }
                }
            ]
        }"#;

        let notes = parse_raw_output(raw).unwrap();
        assert_eq!(notes.len(), 1);
        let source = notes[0].source.as_ref().unwrap();
        assert_eq!(source.url.as_deref(), Some("https://example.org"));
        assert_eq!(source.title.as_deref(), Some("Article"));
        assert_eq!(source.excerpt.as_deref(), Some("Relevant snippet"));
    }
}
