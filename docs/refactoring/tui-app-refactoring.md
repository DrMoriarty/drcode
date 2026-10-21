# Рефакторинг `src/drcode/tui/app.py`

## 1. Проблема

`src/drcode/tui/app.py` — единый файл **4704 строки / ~212 КB**, внутри которого
содержится как модульный код (виджеты, провайдеры команд), так и сам класс
приложения `DrCodeApp` (~3477 строк). Файл выполняет сразу несколько
независимых обязанностей, что делает его трудным для чтения, ревью и тестирования.

Цель — разбить файл на логически обособленные модули без изменения поведения:
виджеты и провайдеры выносятся в свои файлы, а класс `DrCodeApp` делится на
слои (оболочка / отрисовка чата / конвейер / действия).

## 2. Текущая структура файла (диаграмма)

| Диапазон (строки) | содержимое | класс/функция |
|---|---|---|
| 1–164 | импорты + модульные константы | `_SPINNER_FRAMES`, `NETWORK_RETRIES`, `NETWORK_DELAY_MAX` и др. |
| 165–181 | хелперы сетевых retry | `_is_retryable_network_error`, `_network_retry_delay` |
| 183–236 | учёт токенов | `SessionTokenUsage` |
| 219–271 | сайдбар | `TaskItem`, `Sidebar` |
| 273–400 | поля ввода | `WordNavTextArea`, `NewTaskInput`, `ChatInput` |
| 400–811 | блоки чата | `UserBlock`, `ReasoningBlock`, `AssistantBlock`, `ToolCallBlock` + `_render_body`, `_markdown_text`, `_streamable_markdown`, `_split`, `_primary_arg_suffix` |
| 813–1225 | палитра команд | `SpacedCommandList`, `SettingsPalette`, `Service/ModelConfig/ProjectSettings/RetentionOptions/FinalizationOptions/ThemeOptions/StallNudgeOptionsProvider`, `_color_spec`, `_theme_preview`, `spec` |
| 1227–4704 | класс приложения | `DrCodeApp` (~3477 строк) + `run_app` |

## 3. Публичный и внутренний surface, который надо сохранить

Тесты и публичный API напрямую зависят от этих симвлов — рефакторинг не должен
их ломать (либо обновить зависимости, либо сохранить):

- Публично: `from drcode.tui.app import run_app` (`src/drcode/__init__.py:100`),
  `DrCodeApp`, `TaskItem`, `AssistantBlock`, `ReasoningBlock` (`tests/`).
- Внутренне (используется в `tests/test_network_app.py`, `tests/test_stream_layout.py`):
  `app._advance`, `app._check_model`, `app._write_chat`, `app.agent_running`,
  `app.tester_running`, `app.viewed_id`, `app._new_task_mode`, `app.new_task_kind`,
  `app._agent_failed`, `app._tester_retry_at`, `app._tester_fail_count`,
  `app._render_header_tall`, `app._render_activity`, `app._render_model_status`.

## 4. Предлагаемая декомпозиция

### Этап 0 — вынос модульного кода (низориск, сразу даёт ~1000 строк)

| Новый модуль | Переносимый код (строки) | Примечание |
|---|---|---|
| `tui/usage.py` | 141–236: константы, `_is_retryable_network_error`, `_network_retry_delay`, `SessionTokenUsage` | утилиты |
| `tui/inputs.py` | 273–400: `WordNavTextArea`, `NewTaskInput`, `ChatInput` | виджеты ввода |
| `tui/blocks.py` | 400–811: `UserBlock`, `ReasoningBlock`, `AssistantBlock`, `ToolCallBlock` + `_render_body`, `_markdown_text`, `_streamable_markdown`, `_split`, `_primary_arg_suffix` | **тесты импортируют `AssistantBlock`/`ReasoningBlock`** → `app.py` должен ре-экспортировать их |
| `tui/sidebar.py` | 219–271: `TaskItem`, `Sidebar` | виджеты |
| `tui/command_providers.py` | 813–1225: `SpacedCommandList`, `SettingsPalette` + все `Provider`/`CommandPalette` + `_color_spec`, `_theme_preview`, `spec` | |

Импорты из `..agent` (большой блок 80–104) стоит разбить по модулям: переносить
импортируемые хелперы (`render_history_for_summary`, `compacted_messages`,
`run_context_tokens`, `last_response_usage`, `probe_model` и т.д.) ближе к
коду, который их использует (конвейер / блоки), чтобы не держать гигантский
блок импортов в `app.py`.

### Этап 1 — деление класса `DrCodeApp` на слои

Внутри класса методы естественно кластеризуются. Предлагается вынести:

1. **`tui/chat_view.py`** — слой отрисовки/обновления чата и статусов:
   `view_task`, `_render_task_history`, `_render_activity`,
   `_render_model_status`, `_render_header_tall`, `_render_body`, `_write_chat`,
   `refresh_project_info`, `refresh_sidebar`, `toggle_sidebar_section`,
   `refresh_connection`, `refresh_status`, `refresh_task_status`.
2. **`tui/pipeline.py`** — конвейер исполнения: `run_agent` (~583 стр., самый
   крупный метод — 3220–3803), `run_tester` (~294 стр., 3803–4097), `_advance`,
   `_check_model`, bookkeeping (`agent_running`, `tester_running`,
   `_agent_failed`, `_tester_retry_at`, `_tester_fail_count`).
3. **`tui/actions.py`** — все `action_*`-обработчики клавиш +
   `submit_chat_input`, `submit_new_task`, `cancel_new_task`,
   `toggle_new_task_kind`, хелперы нового задачника (`_new_task_mode`,
   `new_task_kind`, `_KIND_CYCLE`).
4. **`tui/app.py`** — оставшийся «скел»: `compose`, `on_mount`, `on_unmount`,
   раскладка виджетов (`chat_log`, `chat_input`, `sidebar`, `project_info`,
   `model_status`, `activity_status`), `viewed`, `check_action`,
   `get_system_commands`, привязки, `run_app`, переписанные импорты.

> **Важно:** `tests/test_network_app.py` вызывает `net_app._advance()`,
> `net_app._check_model()`, читает `net_app.agent_running`,
> `net_app._agent_failed`, `net_app._tester_retry_at`, `net_app._tester_fail_count`;
> `tests/test_stream_layout.py` вызывает `app._write_chat(block)`. Значит, либо
> оставляем эти члены как тонкие делегирующие методы на `DrCodeApp`, либо
> обновляем импорты/вызовы в этих двух тестах. Рекомендация: вынести в
> `pipeline.py`/`chat_view.py` как функции-методы, принимающие `app`/`manager`,
> и обновить тесты (это часть объёма задачи).

## 5. Порядок выполнения (низориск → высоорик)

1. Этап 0: вынос модульного кода в 5 новых файлов; `app.py` ре-экспортирует
   публичные симвлы. Прогон `ruff check src` + `pyright`.
2. Этап 1a: вынести `chat_view.py` (отрисовка) — низорик, не трогает конвейер.
3. Этап 1b: вынести `actions.py` (об处理чики) — средний риск.
4. Этап 1c: вынести `pipeline.py` (`run_agent`/`run_tester`/`_advance`) — высокий
   риск, самый связный код.
5. Обновить `tests/test_network_app.py`, `tests/test_stream_layout.py` под
   новые модули; финальный прогон `ruff` + `pyright` + ручной запущен TUI.

## 6. Риски и оговорки

- **Круговые импорты**: новые `tui.*`-модули должны импортировать ядровые
  символы из `..` (пакет `drcode`), а не из `app.py`, чтобы не создать
  `app → submodule → app` cycle.
- **Сохранение surface**: публичные (`run_app`, `DrCodeApp`) и внутренние
  (`_advance`, `_write_chat`, `_check_model`, флаги состояния) члены должны
  оставаться доступными — либо через ре-экспорт, либо через обновление тестов.
- **Right-size блоков**: `blocks.py`/`command_providers.py` — самые «чистые»
  кандидаты; `pipeline.py` — самый связный, требует внимательного выделения
  зависимостей от состояния приложения.
- **Верификация**: без CI — только `uv run ruff check src`, `uv run pyright` и
  ручной прогон `uv run drcode` в scratch-каталоге.

## 7. Ожидаемый результат

- `app.py` уменьшается с ~3477 строк класса до ~1500–2000 (оболочка + конвейер-
  делегаты); 5–6 новых сфокусированных модулей под виджеты/провайдеры/слои.
- Каждый модуль имеет单一 ответственность, легче ревьюится и покрывается
  тестами; падуются барьеры для внесения следующих изменений.
