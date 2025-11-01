"""Dialog for AI-assisted flashcard generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import aqt.forms
from anki import ai_generation_pb2 as ai_pb
from anki.collection import AddNoteRequest
from anki.decks import DeckId
from anki.errors import AnkiError
from anki.notes import Note
from anki.utils import strip_html
from aqt import AnkiQt
from aqt.operations import CollectionOp, QueryOp
from aqt.qt import *
from aqt.utils import (
    openFile,
    restoreGeom,
    saveGeom,
    showException,
    showInfo,
    tooltip,
    tr,
)

PROVIDER_ORDER = (
    ai_pb.Provider.PROVIDER_GEMINI,
    ai_pb.Provider.PROVIDER_OPENAI,
    ai_pb.Provider.PROVIDER_OPENROUTER,
    ai_pb.Provider.PROVIDER_PERPLEXITY,
)

DEFAULT_MODELS: dict[int, str] = {
    ai_pb.Provider.PROVIDER_GEMINI: "gemini-1.5-flash",
    ai_pb.Provider.PROVIDER_OPENAI: "gpt-4o-mini",
    ai_pb.Provider.PROVIDER_OPENROUTER: "openrouter/anthropic/claude-3.5-sonnet",
    ai_pb.Provider.PROVIDER_PERPLEXITY: "sonar-reasoning",
}


@dataclass
class PreviewNote:
    proto: ai_pb.GeneratedNote
    display_front: str
    display_back: str
    display_source: str


class AiGeneratorDialog(QDialog):
    def __init__(
        self,
        mw: AnkiQt,
        *,
        note_type_id: Optional[int] = None,
        deck_id: Optional[int] = None,
    ) -> None:
        super().__init__(mw, Qt.WindowType.Window)
        self.mw = mw
        self.form = aqt.forms.ai_generator.Ui_AiGeneratorDialog()
        self.form.setupUi(self)

        self._generated_notes: List[PreviewNote] = []
        self._provider_masked: Dict[int, bool] = {}
        self._note_type_ids: Dict[int, int] = {}
        self._deck_ids: Dict[int, int] = {}
        self._models_cache: Dict[int, List[str]] = {}
        self._selected_models: Dict[int, str] = {}
        self._file_path: Optional[Path] = None
        self._requested_note_type_id = note_type_id
        self._requested_deck_id = deck_id

        self._setup_ui()
        self._load_configuration()
        self._load_note_types()
        self._load_decks()

        restoreGeom(self, "aiGenerator")
        self.show()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle(tr.ai_generation_window_title())

        tabs = self.form.inputTabs
        tabs.setTabText(tabs.indexOf(self.form.textTab), tr.ai_generation_tab_text())
        tabs.setTabText(tabs.indexOf(self.form.urlTab), tr.ai_generation_tab_url())
        tabs.setTabText(tabs.indexOf(self.form.fileTab), tr.ai_generation_tab_file())

        self.form.textInput.setPlaceholderText(tr.ai_generation_text_placeholder())
        self.form.urlInput.setPlaceholderText(tr.ai_generation_url_placeholder())
        self.form.urlPreview.setPlaceholderText(
            tr.ai_generation_url_preview_placeholder()
        )
        self.form.filePathInput.setPlaceholderText(tr.ai_generation_file_placeholder())
        self.form.browseButton.setText(tr.ai_generation_browse_button())
        self.form.fetchButton.setText(tr.ai_generation_fetch_button())
        self.form.fileHint.setText(tr.ai_generation_file_hint())

        self.form.configGroup.setTitle(tr.ai_generation_configuration_group())
        self.form.providerLabel.setText(tr.ai_generation_provider_label())
        self.form.geminiLabel.setText(tr.ai_generation_gemini_key_label())
        self.form.openrouterLabel.setText(tr.ai_generation_openrouter_key_label())
        self.form.openaiLabel.setText(tr.ai_generation_openai_key_label())
        self.form.perplexityLabel.setText(tr.ai_generation_perplexity_key_label())
        self.form.noteTypeLabel.setText(tr.ai_generation_note_type_label())
        self.form.useDefaultNoteType.setText(tr.ai_generation_use_default_notetype())
        self.form.deckLabel.setText(tr.ai_generation_deck_label())
        self.form.maxCardsLabel.setText(tr.ai_generation_max_cards_label())
        self.form.modelLabel.setText(tr.ai_generation_model_override_label())
        self.form.modelRefreshButton.setText(tr.ai_generation_model_refresh_button())
        self.form.promptLabel.setText(tr.ai_generation_prompt_override_label())

        self.form.generateButton.setText(tr.ai_generation_generate_button())
        self.form.clearButton.setText(tr.ai_generation_clear_button())
        self.form.addSelectedButton.setText(tr.ai_generation_add_selected_button())
        self.form.addAllButton.setText(tr.ai_generation_add_all_button())

        self.form.previewGroup.setTitle(tr.ai_generation_preview_group())

        self.form.previewTree.setColumnCount(3)
        self.form.previewTree.setHeaderLabels(
            [
                tr.ai_generation_preview_column_front(),
                tr.ai_generation_preview_column_back(),
                tr.ai_generation_preview_column_source(),
            ]
        )
        self.form.previewTree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.form.previewTree.itemSelectionChanged.connect(self._update_add_buttons)

        self.form.generateButton.clicked.connect(self._on_generate_clicked)
        self.form.clearButton.clicked.connect(self._on_clear_clicked)
        self.form.addAllButton.clicked.connect(self._on_add_all)
        self.form.addSelectedButton.clicked.connect(self._on_add_selected)
        self.form.buttonBox.rejected.connect(self.reject)

        self.form.providerCombo.currentIndexChanged.connect(self._on_provider_changed)
        self.form.useDefaultNoteType.toggled.connect(self._on_default_notetype_toggled)
        self.form.browseButton.clicked.connect(self._on_browse_file)
        self.form.fetchButton.setVisible(False)
        self.form.urlPreview.setVisible(False)

        self.form.maxCardsSpin.setMinimum(1)
        self.form.maxCardsSpin.setMaximum(50)

        self.form.progressBar.setValue(0)
        self.form.progressBar.setTextVisible(False)

        combo = self.form.modelOverrideCombo
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setDuplicatesEnabled(False)
        combo.setEditable(True)
        combo.lineEdit().setClearButtonEnabled(True)
        combo.editTextChanged.connect(self._on_model_text_changed)

        self._api_key_inputs = {
            ai_pb.Provider.PROVIDER_GEMINI: self.form.geminiKey,
            ai_pb.Provider.PROVIDER_OPENROUTER: self.form.openrouterKey,
            ai_pb.Provider.PROVIDER_OPENAI: self.form.openaiKey,
            ai_pb.Provider.PROVIDER_PERPLEXITY: self.form.perplexityKey,
        }
        for widget in self._api_key_inputs.values():
            widget.setPlaceholderText(tr.ai_generation_api_key_placeholder())

        for provider in PROVIDER_ORDER:
            self.form.providerCombo.addItem(self._provider_label(provider), provider)

        self.form.modelRefreshButton.clicked.connect(self._on_refresh_models)
        self._update_add_buttons()
        self._set_model_placeholder(DEFAULT_MODELS.get(self.current_provider(), ""))

    def _provider_label(self, provider: int) -> str:
        if provider == ai_pb.Provider.PROVIDER_GEMINI:
            return tr.ai_generation_provider_gemini()
        if provider == ai_pb.Provider.PROVIDER_OPENROUTER:
            return tr.ai_generation_provider_openrouter()
        if provider == ai_pb.Provider.PROVIDER_OPENAI:
            return tr.ai_generation_provider_openai()
        if provider == ai_pb.Provider.PROVIDER_PERPLEXITY:
            return tr.ai_generation_provider_perplexity()
        return "?"

    def _set_model_placeholder(self, placeholder: str) -> None:
        line_edit = self.form.modelOverrideCombo.lineEdit()
        if line_edit:
            line_edit.setPlaceholderText(placeholder)

    def _populate_model_combo(
        self,
        provider: int,
        models: Optional[Iterable[str]] = None,
        *,
        preserve_current: bool = True,
    ) -> None:
        combo = self.form.modelOverrideCombo
        current_text = (
            combo.currentText()
            if preserve_current
            else self._selected_models.get(provider, "")
        )
        source = models if models is not None else self._models_cache.get(provider, [])
        unique: List[str] = []
        seen: set[str] = set()
        for model in source or []:
            trimmed = model.strip()
            if trimmed and trimmed not in seen:
                unique.append(trimmed)
                seen.add(trimmed)

        combo.blockSignals(True)
        combo.clear()
        for model in unique:
            combo.addItem(model)
        combo.setEditText(current_text or "")
        combo.blockSignals(False)

        if current_text.strip():
            self._selected_models[provider] = current_text.strip()

        self._set_model_placeholder(DEFAULT_MODELS.get(provider, ""))

    def _on_model_text_changed(self, text: str) -> None:
        provider = self.current_provider()
        stripped = text.strip()
        if stripped:
            self._selected_models[provider] = stripped
        else:
            self._selected_models.pop(provider, None)

    def _on_refresh_models(self) -> None:
        provider = self.current_provider()
        request = ai_pb.ListModelsRequest()
        request.provider = provider

        api_input = self._api_key_inputs.get(provider)
        if api_input:
            request.api_key = api_input.text().strip()

        combo = self.form.modelOverrideCombo
        button = self.form.modelRefreshButton

        combo.setEnabled(False)
        button.setEnabled(False)

        def restore_ui() -> None:
            combo.setEnabled(True)
            button.setEnabled(True)

        def on_success(response: ai_pb.ListModelsResponse) -> None:
            restore_ui()
            models = list(response.models)
            self._models_cache[provider] = models
            self._populate_model_combo(provider, models, preserve_current=True)
            if not combo.currentText().strip() and models:
                combo.setEditText(models[0])
            self._set_model_placeholder(DEFAULT_MODELS.get(provider, ""))

        def on_failure(error: Exception) -> None:
            restore_ui()
            showException(parent=self, exception=error)

        QueryOp(
            parent=self,
            op=lambda col: col.list_ai_models(request),
            success=on_success,
        ).failure(on_failure).run_in_background()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_configuration(self) -> None:
        config = self.mw.col.get_ai_generation_config()

        selected_provider = config.selected_provider
        if selected_provider not in PROVIDER_ORDER:
            selected_provider = ai_pb.Provider.PROVIDER_GEMINI

        self._selected_models[selected_provider] = config.preferred_model
        index = self.form.providerCombo.findData(selected_provider)
        if index != -1:
            self.form.providerCombo.setCurrentIndex(index)

        if config.default_max_cards:
            self.form.maxCardsSpin.setValue(config.default_max_cards)

        self.form.modelOverrideCombo.setEditText(config.preferred_model)
        self.form.promptOverrideEdit.setPlainText("")

        self._provider_masked.clear()
        for entry in config.api_keys:
            self._provider_masked[entry.provider] = entry.masked
            line_edit = self._api_key_inputs.get(entry.provider)
            if not line_edit:
                continue
            if entry.masked:
                line_edit.clear()
                line_edit.setPlaceholderText(
                    tr.ai_generation_saved_api_key_placeholder()
                )
            else:
                line_edit.setText(entry.api_key)

        default_ntid = config.default_note_type_id
        self._default_notetype_id = default_ntid if default_ntid else 0
        self.form.useDefaultNoteType.setChecked(self._default_notetype_id > 0)
        self._on_default_notetype_toggled(self.form.useDefaultNoteType.isChecked())
        if self._requested_note_type_id:
            self.form.useDefaultNoteType.setChecked(False)

    def _load_note_types(self) -> None:
        self.form.noteTypeCombo.clear()
        self._note_type_ids.clear()
        note_types = self.mw.col.models.all_names_and_ids()
        for idx, entry in enumerate(note_types):
            self.form.noteTypeCombo.addItem(entry.name, entry.id)
            self._note_type_ids[idx] = entry.id

        # choose default: current note type or saved default
        current_ntid = self.mw.col.models.current()["id"]
        target_ntid = (
            self._requested_note_type_id
            or (
                self._default_notetype_id
                if self.form.useDefaultNoteType.isChecked()
                else 0
            )
            or current_ntid
        )
        combo_index = self.form.noteTypeCombo.findData(target_ntid)
        if combo_index != -1:
            self.form.noteTypeCombo.setCurrentIndex(combo_index)

    def _load_decks(self) -> None:
        self.form.deckCombo.clear()
        self._deck_ids.clear()
        decks = self.mw.col.decks.all_names_and_ids()
        for idx, entry in enumerate(decks):
            self.form.deckCombo.addItem(entry.name, entry.id)
            self._deck_ids[idx] = entry.id

        current_deck = int(
            self._requested_deck_id or self.mw.col.decks.get_current_id()
        )
        combo_index = self.form.deckCombo.findData(current_deck)
        if combo_index != -1:
            self.form.deckCombo.setCurrentIndex(combo_index)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_provider_changed(self) -> None:
        provider = self.current_provider()
        self._set_model_placeholder(DEFAULT_MODELS.get(provider, ""))
        self._populate_model_combo(provider, preserve_current=False)

    def _on_default_notetype_toggled(self, checked: bool) -> None:
        self.form.noteTypeCombo.setEnabled(not checked)

    def _on_browse_file(self) -> None:
        path = openFile(
            parent=self,
            caption=tr.ai_generation_select_file_dialog_title(),
            dir=str(Path.home()),
            filter="PDF (*.pdf);;Text (*.txt *.md *.markdown *.rtf);;All Files (*)",
        )
        if path:
            self._file_path = Path(path)
            self.form.filePathInput.setText(str(self._file_path))

    def _on_generate_clicked(self) -> None:
        try:
            request = self._build_request()
            config_request = self._build_config_request()
        except AnkiError as exc:
            showInfo(str(exc), parent=self)
            return

        self._set_busy(True)

        def op(col) -> ai_pb.GenerateFlashcardsResponse:
            if config_request:
                col.set_ai_generation_config(config_request)
            return col.generate_flashcards(request)

        (
            QueryOp(parent=self, op=op, success=self._on_generate_finished)
            .failure(self._on_generate_failed)
            .with_progress(tr.ai_generation_progress_generating())
        ).run_in_background()

    def _on_generate_finished(self, response: ai_pb.GenerateFlashcardsResponse) -> None:
        self._set_busy(False)
        self._populate_preview(response)
        self._reset_api_key_fields()

        if not response.notes:
            showInfo(tr.ai_generation_no_cards_returned(), parent=self)

    def _on_generate_failed(self, error: Exception) -> None:
        self._set_busy(False)
        showException(parent=self, exception=error)

    def _on_add_all(self) -> None:
        self._add_notes(range(len(self._generated_notes)))

    def _on_add_selected(self) -> None:
        indices = []
        for item in self.form.previewTree.selectedItems():
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(idx, int):
                indices.append(idx)
        if indices:
            self._add_notes(indices)

    def _on_clear_clicked(self) -> None:
        self.form.textInput.clear()
        self.form.urlInput.clear()
        self.form.urlPreview.clear()
        self.form.filePathInput.clear()
        provider = self.current_provider()
        self.form.modelOverrideCombo.setEditText("")
        self._selected_models.pop(provider, None)
        self.form.promptOverrideEdit.clear()
        self._file_path = None
        self._set_preview_notes([])

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        saveGeom(self, "aiGenerator")
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def current_provider(self) -> int:
        return int(self.form.providerCombo.currentData())

    def _build_request(self) -> ai_pb.GenerateFlashcardsRequest:
        request = ai_pb.GenerateFlashcardsRequest()
        provider = self.current_provider()
        request.provider = provider

        current_tab = self.form.inputTabs.currentWidget()
        if current_tab is self.form.textTab:
            text = self.form.textInput.toPlainText().strip()
            if not text:
                raise AnkiError(tr.ai_generation_error_empty_text())
            request.input_type = ai_pb.InputType.INPUT_TYPE_TEXT
            request.text = text
        elif current_tab is self.form.urlTab:
            url = self.form.urlInput.text().strip()
            if not url:
                raise AnkiError(tr.ai_generation_error_empty_url())
            request.input_type = ai_pb.InputType.INPUT_TYPE_URL
            request.url = url
        else:
            if not self._file_path:
                raise AnkiError(tr.ai_generation_error_missing_file())
            data = self._file_path.read_bytes()
            if not data:
                raise AnkiError(tr.ai_generation_error_empty_file())
            file_input = ai_pb.FileInput()
            file_input.filename = self._file_path.name
            file_input.data = data
            request.input_type = ai_pb.InputType.INPUT_TYPE_FILE
            request.file.CopyFrom(file_input)

        request.max_cards = self.form.maxCardsSpin.value()

        if not self.form.useDefaultNoteType.isChecked():
            ntid = self._current_note_type_id()
            if ntid:
                request.note_type_id = ntid
        request.use_default_note_type = self.form.useDefaultNoteType.isChecked()

        deck_id = self._current_deck_id()
        if deck_id:
            request.deck_id = deck_id

        model_override = self.form.modelOverrideCombo.currentText().strip()
        if model_override:
            request.model_override = model_override
            self._selected_models[provider] = model_override
        else:
            self._selected_models.pop(provider, None)

        prompt_override = self.form.promptOverrideEdit.toPlainText().strip()
        if prompt_override:
            request.prompt_override = prompt_override

        return request

    def _build_config_request(self) -> ai_pb.SetAiConfigRequest:
        config = ai_pb.AiGenerationConfig()
        config.selected_provider = self.current_provider()
        if self.form.useDefaultNoteType.isChecked():
            default_ntid = self._current_note_type_id() or self._default_notetype_id
        else:
            default_ntid = 0
        self._default_notetype_id = default_ntid
        config.default_note_type_id = default_ntid
        model_text = self.form.modelOverrideCombo.currentText().strip()
        config.preferred_model = model_text
        if model_text:
            self._selected_models[config.selected_provider] = model_text
        else:
            self._selected_models.pop(config.selected_provider, None)
        config.default_max_cards = self.form.maxCardsSpin.value()

        config.api_keys.extend(self._collect_api_keys())

        request = ai_pb.SetAiConfigRequest()
        request.config.CopyFrom(config)
        request.persist_api_keys = True
        return request

    def _collect_api_keys(self) -> Iterable[ai_pb.ProviderApiKey]:
        for provider, line_edit in self._api_key_inputs.items():
            text = line_edit.text().strip()
            if text:
                yield ai_pb.ProviderApiKey(
                    provider=provider,
                    api_key=text,
                    masked=False,
                )
            elif self._provider_masked.get(provider, False):
                yield ai_pb.ProviderApiKey(provider=provider, masked=True)
            else:
                yield ai_pb.ProviderApiKey(provider=provider, api_key="", masked=False)

    def _reset_api_key_fields(self) -> None:
        for provider, widget in self._api_key_inputs.items():
            text = widget.text().strip()
            if text:
                self._provider_masked[provider] = True
                widget.clear()
                widget.setPlaceholderText(tr.ai_generation_saved_api_key_placeholder())

    def _populate_preview(self, response: ai_pb.GenerateFlashcardsResponse) -> None:
        preview_notes = []
        for proto in response.notes:
            preview_notes.append(
                PreviewNote(
                    proto=proto,
                    display_front=self._display_field(
                        proto, ["front", "question", "prompt"]
                    ),
                    display_back=self._display_field(
                        proto, ["back", "answer", "response"]
                    ),
                    display_source=self._display_source(proto.source),
                )
            )
        self._set_preview_notes(preview_notes)

    def _display_field(
        self, proto: ai_pb.GeneratedNote, preferred_names: List[str]
    ) -> str:
        mapping = {field.name.lower(): field.value for field in proto.fields}
        for name in preferred_names:
            if name in mapping:
                return strip_html(mapping[name])
        if proto.fields:
            return strip_html(proto.fields[0].value)
        return ""

    def _display_source(self, source: Optional[ai_pb.GeneratedNoteSource]) -> str:
        if not source:
            return ""

        parts: List[str] = []
        if source.title:
            parts.append(source.title)
        if source.excerpt:
            parts.append(source.excerpt)
        if source.url:
            parts.append(source.url)
        return "\n".join(parts)

    def _set_preview_notes(self, notes: List[PreviewNote]) -> None:
        self._generated_notes = list(notes)
        tree = self.form.previewTree
        tree.clear()
        for idx, note in enumerate(notes):
            item = QTreeWidgetItem(tree)
            item.setText(0, note.display_front)
            item.setText(1, note.display_back)
            item.setText(2, note.display_source)
            item.setData(0, Qt.ItemDataRole.UserRole, idx)
        self._update_add_buttons()

    def _update_add_buttons(self) -> None:
        has_notes = bool(self._generated_notes)
        self.form.addAllButton.setEnabled(has_notes)
        self.form.addSelectedButton.setEnabled(
            has_notes and bool(self.form.previewTree.selectedItems())
        )

    def _set_busy(self, busy: bool) -> None:
        self.form.generateButton.setDisabled(busy)
        self.form.progressBar.setRange(0, 0 if busy else 1)
        if not busy:
            self.form.progressBar.setValue(0)

    # ------------------------------------------------------------------
    # Note creation
    # ------------------------------------------------------------------

    def _add_notes(self, indices: Iterable[int]) -> None:
        unique_indices = sorted(set(indices), reverse=True)
        if not unique_indices:
            return

        requests: List[AddNoteRequest] = []
        for index in unique_indices:
            if index >= len(self._generated_notes):
                continue
            proto = self._generated_notes[index].proto
            try:
                note, deck_id = self._create_note(proto)
            except AnkiError as err:
                showInfo(str(err), parent=self)
                return
            requests.append(AddNoteRequest(note=note, deck_id=deck_id))

        def op(col):
            return col.add_notes(requests)

        (
            CollectionOp(
                parent=self,
                op=op,
                success=lambda _: self._on_add_success(unique_indices),
            ).failure(self._on_add_failure)
        ).run_in_background(initiator=self)

    def _create_note(self, proto: ai_pb.GeneratedNote) -> tuple[Note, DeckId]:
        notetype_id = proto.note_type_id or self._effective_note_type_id()
        if not notetype_id:
            raise AnkiError(tr.ai_generation_error_missing_notetype())

        deck_id = proto.deck_id or self._current_deck_id()
        if not deck_id:
            deck_id = int(self.mw.col.decks.get_current_id())

        notetype = self.mw.col.models.get(notetype_id)
        if not notetype:
            raise AnkiError(tr.ai_generation_error_invalid_notetype())

        note = self.mw.col.new_note(notetype)

        field_map = {field.name.lower(): field.value for field in proto.fields}
        for idx, field in enumerate(notetype["flds"]):
            name = field["name"].lower()
            value = field_map.pop(name, "")
            note.fields[idx] = value

        if proto.source:
            source_value = self._format_source(proto.source)
            self._assign_source_field(note, notetype, source_value)

        # Append any remaining fields to the back field if present
        if field_map:
            back_index = next(
                (
                    idx
                    for idx, field in enumerate(notetype["flds"])
                    if field["name"].lower() == "back"
                ),
                None,
            )
            if back_index is not None:
                extras = [
                    value.strip() for value in field_map.values() if value.strip()
                ]
                if extras:
                    segments = [note.fields[back_index].strip(), "\n\n".join(extras)]
                    note.fields[back_index] = "\n\n".join(
                        segment for segment in segments if segment
                    )

        return note, DeckId(deck_id)

    def _assign_source_field(self, note: Note, notetype: dict, value: str) -> None:
        for idx, field in enumerate(notetype["flds"]):
            if field["name"].lower() == "source":
                note.fields[idx] = value
                return
        # fallback: append to last field if Source not found
        note.fields[-1] = (note.fields[-1] + "\n\n" + value).strip()

    def _format_source(self, source: ai_pb.GeneratedNoteSource) -> str:
        parts: List[str] = []
        if source.excerpt:
            parts.append(escape(source.excerpt))
        if source.url:
            if source.title:
                parts.append(
                    f'<a href="{escape(source.url, quote=True)}">'
                    f"{escape(source.title)}</a>"
                )
            else:
                parts.append(escape(source.url))
        elif source.title:
            parts.append(escape(source.title))
        return "<br>".join(parts)

    def _on_add_success(self, indices: Iterable[int]) -> None:
        indices_set = set(indices)
        self._generated_notes = [
            note
            for idx, note in enumerate(self._generated_notes)
            if idx not in indices_set
        ]
        self._set_preview_notes(self._generated_notes)
        tooltip(tr.ai_generation_added_cards_tooltip(), parent=self)

    def _on_add_failure(self, error: Exception) -> None:
        showException(parent=self, exception=error)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_note_type_id(self) -> int:
        if self.form.useDefaultNoteType.isChecked() and self._default_notetype_id:
            return self._default_notetype_id
        return self._current_note_type_id()

    def _current_note_type_id(self) -> int:
        return int(self.form.noteTypeCombo.currentData() or 0)

    def _current_deck_id(self) -> int:
        return int(self.form.deckCombo.currentData() or 0)


def open_ai_generator_dialog(
    mw: AnkiQt, *, note_type_id: Optional[int] = None, deck_id: Optional[int] = None
) -> AiGeneratorDialog:
    return AiGeneratorDialog(mw, note_type_id=note_type_id, deck_id=deck_id)
