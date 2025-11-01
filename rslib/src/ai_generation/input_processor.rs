use std::path::Path;

use pdf_extract::extract_text_from_mem;
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Url};
use scraper::Html;

use crate::ai_generation::{AiResult, FilePayload, GenerationRequest, InputPayload};

#[derive(Debug, Clone, Default)]
pub struct ProcessedInput {
    pub text: String,
    pub source_url: Option<String>,
    pub file: Option<FilePayload>,
}

pub struct InputProcessor;

impl InputProcessor {
    pub async fn prepare(request: &GenerationRequest) -> AiResult<ProcessedInput> {
        let mut processed = match &request.input {
            InputPayload::Text(text) => Self::process_text(text),
            InputPayload::Url(url) => Self::process_url(url).await?,
            InputPayload::File(file) => Self::process_file(file)?,
        };

        if processed.text.trim().is_empty() {
            crate::invalid_input!("input did not contain any readable content");
        }

        // ensure any excessive whitespace is trimmed after validation
        processed.text = normalize_whitespace(&processed.text);

        Ok(processed)
    }

    fn process_text(text: &str) -> ProcessedInput {
        ProcessedInput {
            text: text.to_owned(),
            source_url: None,
            file: None,
        }
    }

    async fn process_url(url: &str) -> AiResult<ProcessedInput> {
        let client = Client::new();
        let response = client.get(url).send().await?.error_for_status()?;
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_ascii_lowercase());
        let bytes = response.bytes().await?.to_vec();

        if is_pdf_mime(content_type.as_deref(), Some(url)) {
            let text = extract_pdf_text(&bytes)?;
            let filename = filename_from_url(url).unwrap_or_else(|| url.to_string());
            let payload = FilePayload::new(filename, bytes, Some("application/pdf".to_string()));
            Ok(ProcessedInput {
                text,
                source_url: Some(url.to_string()),
                file: Some(payload),
            })
        } else {
            let decoded = decode_text(&bytes);
            let content = if is_html_mime(content_type.as_deref()) {
                html_to_text(&decoded)
            } else {
                decoded
            };

            Ok(ProcessedInput {
                text: content,
                source_url: Some(url.to_string()),
                file: None,
            })
        }
    }

    fn process_file(file: &FilePayload) -> AiResult<ProcessedInput> {
        let mime = infer_mime(file);

        if is_pdf_mime(mime.as_deref(), None) {
            let text = extract_pdf_text(&file.data)?;
            Ok(ProcessedInput {
                text,
                source_url: None,
                file: Some(file.clone()),
            })
        } else {
            let decoded = decode_text(&file.data);
            let content = if is_html_mime(mime.as_deref()) {
                html_to_text(&decoded)
            } else {
                decoded
            };

            Ok(ProcessedInput {
                text: content,
                source_url: None,
                file: Some(file.clone()),
            })
        }
    }
}

fn decode_text(data: &[u8]) -> String {
    match String::from_utf8(data.to_vec()) {
        Ok(text) => text,
        Err(err) => String::from_utf8_lossy(err.as_bytes()).to_string(),
    }
}

fn infer_mime(file: &FilePayload) -> Option<String> {
    if let Some(mimetype) = &file.mimetype {
        return Some(mimetype.to_ascii_lowercase());
    }

    Path::new(&file.filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| match ext.to_ascii_lowercase().as_str() {
            "pdf" => Some("application/pdf"),
            "html" | "htm" => Some("text/html"),
            "md" | "markdown" => Some("text/markdown"),
            "json" => Some("application/json"),
            "csv" => Some("text/csv"),
            "txt" => Some("text/plain"),
            _ => None,
        })
        .map(|s| s.to_string())
}

fn is_pdf_mime(mime: Option<&str>, url: Option<&str>) -> bool {
    if let Some(mime) = mime {
        if mime.contains("pdf") {
            return true;
        }
    }

    url.and_then(|raw| {
        Url::parse(raw).ok().and_then(|parsed| {
            parsed
                .path_segments()
                .and_then(|segments| segments.last().map(|name| name.ends_with(".pdf")))
        })
    })
    .unwrap_or(false)
}

fn is_html_mime(mime: Option<&str>) -> bool {
    mime.map(|value| value.contains("html") || value.starts_with("text/"))
        .unwrap_or(false)
}

fn extract_pdf_text(data: &[u8]) -> AiResult<String> {
    match extract_text_from_mem(data) {
        Ok(text) => Ok(text),
        Err(err) => crate::invalid_input!(err, "unable to read PDF input"),
    }
}

fn html_to_text(html: &str) -> String {
    let document = Html::parse_document(html);
    document
        .root_element()
        .text()
        .map(|fragment| fragment.trim())
        .filter(|fragment| !fragment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn filename_from_url(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.path_segments().and_then(|segments| segments.last().map(|s| s.to_string())))
        .filter(|name| !name.is_empty())
}

fn normalize_whitespace(text: &str) -> String {
    let mut normalized = String::new();
    let mut previous_blank = true;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !previous_blank && !normalized.is_empty() {
                normalized.push_str("\n\n");
            }
            previous_blank = true;
        } else {
            if !normalized.is_empty() && !previous_blank {
                normalized.push('\n');
            }
            normalized.push_str(trimmed);
            previous_blank = false;
        }
    }

    if normalized.is_empty() {
        text.trim().to_string()
    } else {
        normalized
    }
}



