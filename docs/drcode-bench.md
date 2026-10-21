# drcode-bench — план реализации бенчмарка полной связки агентов

Отдельный проект (свой репозиторий), не входит в drcode. drcode подключается как
зависимость (`drcode @ git+...` или path-dep, ставится через `uv`). Цель —
измерять и проверять корректность полного цикла
**coding → testing → compaction** и всех межагентных взаимодействий в условиях,
когда каждому агенту назначена своя модель.

Этот документ описывает один режим запуска: **headless-приложение + симулятор
пользователя, один сценарий = один subprocess**. Отдельный полноэкранный запуск
drcode как внешнего процесса в pty (режим «как в бою») в план не входит.

---

## 1. Что берётся из drcode как есть (без правок исходников)

### 1.1. Разные модели на роль — готовый механизм

- Каталог провайдеров: `~/.config/drcode/models.json` (`src/drcode/providers.py`,
  `UserModelsConfig`) — провайдеры с `base_url`/`api_key`/`health_check_url` и
  списком моделей (`id`, `max_context_tokens`, `max_tokens`).
- Выбор модели на роль: ключ `models` в `<project>/.drcode/drcode.json`:
  ```json
  {
    "models": {
      "coding": "provider-a/qwen3-coder-30b",
      "tester": "provider-b/qwen3-8b",
      "compaction": "provider-b/qwen3-8b"
    }
  }
  ```
- Резолвер `providers.resolve(config, role)`: `DRCODE_MODEL`/`DRCODE_BASE_URL`
  env (все роли сразу!) > project `models[role]` > catalog default > flat config.
- Следствие для бенчмарка: **никогда не выставлять `DRCODE_MODEL`/`DRCODE_BASE_URL`
  в окружении прогона**, иначе env-слой перебьёт per-role выборку для всех ролей.
  Нужные значения задаются только файлами.
- Изоляция каталога: `XDG_CONFIG_HOME` читается при импорте `drcode.config` —
  один сценарий → один subprocess со своим `XDG_CONFIG_HOME` = полностью
  изолированный user-level каталог.

### 1.2. Принудительная компактизация без правок

- Порог — процент от `max_context_tokens`, проверяется в `_fill_over_threshold`
  (`src/drcode/tui/pipeline.py`). **Важно: у тестировщика порог считается от
  конфига CODING-модели** (`_fill_over_threshold` использует `_coding_config()`),
  а не от модели тестера. «Маленький контекст» всегда ставится кодинговой модели,
  даже когда тестируется компакция у тестера.
- Чтобы компакция срабатывала за минуты на коротких задачах:
  `DRCODE_MAX_CONTEXT_TOKENS=2000..4000` (или `max_context_tokens` в записи
  модели каталога — она перекроет flat config для этой роли).
- Компакция у кодера срабатывает двояко: на границе turn'ов (`_maybe_compact`) и
  mid-run на tool-boundary (pause → `sanitized_messages` → `unfinished=True` →
  summary → continuation-turn). У тестера — mid-round
  (`_maybe_compact_tester_round`), раунд продолжается из компактного транскрипта.
- Failure компакции never fatal: пишется event «compaction failed», gate
  отключается для задачи (`_compaction_failed`), ран продолжается несжатым.
  Бенчмарк должен это состояние распознавать и отмечать в отчёте.

### 1.3. Полный цикл «кодер → тестер → баг → кодер» — уже в state machine

- Тестер: `set_verification(success)` — метка passed/failed и завершение задачи;
  `create_bug_report(title, description)` — новая задача `kind=bug`, попадает
  в голову очереди (`TaskManager.queued()` сортирует
  `(bugs newest-first, queue_head, queue_seq)`).
- Возврат на доработки: `TaskManager.send()` на COMPLETED/TESTING ре-queue'ит
  задачу, старая метка верификации сбрасывается.
- Clarification: `ask_clarifying_question` ставит `AWAITING_CLARIFICATION`;
  ответ пользователя через `send()` возвращает задачу в голову очереди
  (`queue_head=True`).
- Автозавершение: кодер закончил без pending input / unfinished → `mark_done`
  → TESTING; research-задачи тестер не проходят (в сценариях не использовать).

### 1.4. Метрики уже персистятся на диске

Каждая задача — `<project>/.drcode/tasks/<id>.json` (`src/drcode/storage.py`):

- `created_at`, `activated_at`, `completed_at` — тайминг фаз.
- `events: list[TaskEvent]` (timestamp + text) — «Context compacted»,
  «pausing the run to compact history», «compaction failed», сетевые ретраи,
  `tester failed`, «Context fill N% — …». Это основной источник фактов
  «что происходило и когда».
- `messages` — сериализованный транскрипт кодера (pydantic-ai `ModelMessage`):
  у каждого `ModelResponse` есть `usage` (prompt/completion tokens) и timestamp;
  summary компакции — синтетическое сообщение с префиксом
  `COMPACTION_SUMMARY_PREFIX`, исходные user-промпты остаются дословно.
- `tester_messages: list[list[...]]` — по одному транскрипту на testing-раунд
  (та же структура; компакция тестера переписывает последний раунд in-place).
- `verification`, `kind`, `unfinished`, `clarification`, `github_issue`.

Пост-анализ делается целиком из этих JSON — drcode менять не нужно.

### 1.5. Чего на диске нет (закрывается инструментацией бенчмарка)

- **Usage компакционного агента не пишется в task** — только агрегируется в
  сессионный счётчик TUI (`_record_session_usage`).
- **Per-request wall-time нигде не хранится** — timestamps сообщений дают лишь
  приблизительную оценку.
- Обе дырки закрывает recording-обёртка вокруг `agent_model.build_model`
  (см. §5) или внешний HTTP-прокси (см. §9).

### 1.6. Ограничение, определившее архитектуру

Pipeline (`run_agent` / `run_tester` / `_advance`) живёт только внутри Textual
App — это `@work`-воркеры слоя `PipelineLayer` (`src/drcode/tui/pipeline.py`).
Headless-CLI (`drcode --bug-report`) только создаёт задачи. Следовательно:
бенчмарк запускает **настоящий `DrCodeApp` headless** через textual
`run_test()` — этот путь уже обкатан тестами (`tests/test_network_app.py`
гоняет реальные воркеры и `_advance` именно так), с той разницей, что вместо
`FunctionModel` — настоящие модели, а вместо секунд — минуты.

---

## 2. Архитектура

```
┌────────────────────────── bench run (оркестратор) ──────────────────────────┐
│                                                                             │
│  для каждой пары (scenario, model-combo):                                   │
│                                                                             │
│  1. подготовка среды (tmp)                                                  │
│  2. subprocess: python -m bench.run_one <scenario> <combo> <run_dir>        │
│  3. сбор артефактов → runs/<run_id>/                                        │
│  4. извлечение метрик → runs/<run_id>/metrics.json                          │
└─────────────────────────────────────────────────────────────────────────────┘

внутри bench.run_one (один процесс = один прогон):

  workspace/            ← сид-репозиторий сценария (copy из scenarios/<s>/seed)
  config-home/          ← XDG_CONFIG_HOME: models.json + drcode.json
  project/.drcode/      ← drcode.json (per-role models), settings.json, tasks/

  async with DrCodeApp(project_root).run_test(size=(120, 40)) as pilot:
      manager.create(...)                    # постановка задач в очередь
      user_sim = asyncio.create_task(...)    # симулятор пользователя
      recorder = patch build_model(...)      # recording-обёртка модели
      await wait_settle(...)                 # нет active/testing, цели выполнены
  → metrics
```

Ключевые принципы:

- **Бенчмарк не копирует и не форкает pipeline drcode.** Крутится реальный
  `DrCodeApp` со всеми воркерами, `_advance`-гейтом, ретраями, watchdog'ом.
- **Один сценарий — один subprocess.** Причины: (а) изоляция
  `XDG_CONFIG_HOME` и `DRCODE_*`-среды — в одном процессе env и импорты
  перемешаются; (б) репозиторный `.env` drcode протекает через
  `pydantic-settings` (`config._REPO_ENV_FILES`) — чистую слоёную конфигурацию
  гарантирует только свежий процесс с вычищенным окружением; (в) параллельные
  прогоны не делят ни файлов, ни event loop.
- **Пользователь — программа.** Все «человеческие» действия (ответы на
  clarification, follow-up-комментарии) выполняет симулятор внутри процесса,
  вызывая `TaskManager.send()` (см. §6).
- **Модель — под наблюдением.** Точный timing/usage даёт recording-обёртка
  вокруг `agent_model.build_model` (см. §5). Это monkeypatch в процессе
  бенчмарка — инструмент наблюдения, не изменение исходников drcode.

---

## 3. Подготовка среды на прогон

Харнесс создаёт временное дерево:

```
<run_tmp>/
├── workspace/                 # корень «проекта» для агентов (cwd для run_bash)
│   ├── <seed-файлы сценария>  # код с багом/заготовкой фичи, свои тесты
│   └── AGENTS.md              # (опция) проектные инструкции для агентов
├── config-home/               # XDG_CONFIG_HOME
│   └── drcode/
│       ├── models.json        # каталог провайдеров текущей комбинации
│       └── drcode.json        # (опция) user-level flat config
└── .git/                      # init + первый commit + bare-remote в tmp
```

Файлы конфигурации генерируются из комбинации матрицы:

- `config-home/drcode/models.json` — провайдеры и модели комбинации
  (формат см. `models.json.example`, поля `Provider`/`ModelEntry`).
- `<workspace>/.drcode/drcode.json` — flat-слой (если нужно) + ключ `models`
  с per-role выборкой.
- `<workspace>/.drcode/settings.json` (`ProjectSettings`):
  - `task_finalization`: `"testing"` (без коммитов) или `"testing_commit"`
    с локальным bare-remote — **никогда не пушить на GitHub**, прогон не должен
    зависеть от внешней сети;
  - `stall_nudge`: `"never"` на время замеров (отдельный сценарий может
    намеренно включить watchdog);
  - `completed_retention`: `"never"` (cleanup на старте не должен трогать
    артефакты).
- Переменные окружения subprocess: `XDG_CONFIG_HOME=<config-home>`,
  все `DRCODE_*` — вычищены, `HOME` можно перенаправить для полной изоляции.

Проверка готовности перед стартом: `providers.resolve()` для каждой роли
должен вернуть ожидаемую модель — простой unit-подобный assert в харнессе до
запуска приложения.

---

## 4. Прогон (bench/run_one.py)

Скелет (псевдокод):

```python
import asyncio
from drcode.tui.app import DrCodeApp
from drcode.task_manager import TaskManager
from drcode.models import TaskKind

async def main(scenario, combo, run_dir):
    install_recorder(run_dir)        # §5: monkeypatch agent_model.build_model
    app = DrCodeApp(Path("workspace"))
    async with app.run_test(size=(120, 40)) as pilot:
        manager = app.manager
        for brief in scenario.briefs:
            manager.create(brief.title, brief.text, kind=brief.kind)
        sim = asyncio.create_task(
            user_simulator(app, scenario, pilot)   # §6
        )
        ok = await wait_settle(app, scenario, pilot, timeout=scenario.timeout)
        sim.cancel()
    if not ok:
        dump_diagnostics(run_dir)
```

Детали:

- `run_test()` у textual работает в headless-режиме и в обычном asyncio-цикле —
  pytest не нужен. `size` влияет только на раскладку.
- Харнесс **не переключает** `viewed_id` (не «открывает» задачи в чате):
  `view_task()` рендерит всю историю и на жирных транскриптах вносит
  рендер-оверхед в замеры. По умолчанию viewed_id пуст — так и оставить.
- **Settle-критерий** (успех прогона): нет задач в ACTIVE/TESTING, все целевые
  задачи сценария COMPLETED, `tester_running=False`, `agent_running=False`,
  и очередь пуста. Плюс scenario-specific check (§7). Таймаут — мягкий предел
  сценария (например, 30–60 мин), по истечении — провал с диагностикой.
- Диагностика при провале: копия `.drcode/`, хвост recording-лога, последние
  events всех задач, `task.unfinished`/`clarification` — чтобы прогон можно
  было разобрать без повторного запуска.

---

## 5. Recording-обёртка модели (bench/recording.py)

Проблема: per-request латентность и usage компакций на диске не сохраняются.
Решение — обернуть точку сборки модели:

```python
import json, time
from drcode import agent_model

class RecordingModel:
    def __init__(self, inner, role_tag, log_path):
        self._inner, self._tag, self._log = inner, role_tag, log_path

    async def request(self, messages, tools, model_settings):  # pydantic-ai Model interface
        t0 = time.monotonic()
        resp = await self._inner.request(messages, tools, model_settings)
        self._log.write(json.dumps({
            "role": self._tag,
            "model": self._inner.model_name,
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "n_history": len(messages),
            "usage": resp.usage() ... ,     # request_tokens / response_tokens
            "ts": time.time(),
        }) + "\n")
        return resp

    # stream_request, __getattr__ — делегация в inner

def install_recorder(log_path):
    orig = agent_model.build_model
    def patched(config):
        inner = orig(config)
        tag = f"{config.model}@{config.base_url}"   # роль видна по выборке
        return RecordingModel(inner, tag, log_path)
    agent_model.build_model = patched
```

Замечания:

- `build_model` вызывается отдельно для каждого агента (кодер, тестер,
  компактизатор) с уже **резолвенным** конфигом роли — тег по
  `config.model`/`base_url` надёжно различает роли (при совпадении модели
  у ролей различать по порядку build-вызовов или расширить обёртку до тега
  роли через место вызова; для матрицы с разными моделями на роль тега
  достаточно).
- Лог — JSONL в `runs/<run_id>/model_log.jsonl`: одна строка = один HTTP-запрос
  к модели. Из него считаются: латентность per request, токены per role,
  число запросов per agent per run.
- Аккуратность с интерфейсом `pydantic_ai.models.Model`: делегировать всё
  через `__getattr__`, переопределив только `request`/`stream_request`.
  Стриминг-путь должен остаться стримингом (TUI потребляет дельты; подмена на
  не-стриминг исказила бы поведение watchdog'а).

Альтернатива/дополнение — внешний записывающий HTTP-прокси (§9): он же даёт
fault injection.

---

## 6. Симулятор пользователя (bench/user_sim.py)

Asyncio-таск в том же event loop, что и приложение. Опрашивает `app.manager`
(поллинг раз в 0.5–1 с достаточно; все мутации — через публичный API
`TaskManager`, который и TUI использует):

1. **Auto-answer clarification.** Задача в `AWAITING_CLARIFICATION` с
   `task.clarification` → симулятор ищет ответ в сценарии
   (`scenario.clarification_answers`: матч по тексту вопроса или по порядку)
   и вызывает `manager.send(task_id, answer)`. Не найден ответ → провал
   сценария (задача зависла).
2. **Follow-up-комментарии.** По расписанию сценария
   (`scenario.followups`: условие = верификация failed / elapsed / событие) —
   `manager.send(task_id, text)`. Главный случай: тестер поставил FAILED —
   симулятор отправляет «вот лог ошибки, исправь» (текст задаётся сценарием).
3. **Защита от залипания.** Если задача ACTIVE, ничего не запущено
   (`agent_running=False`), `unfinished=False` и это длится дольше N секунд —
   это состояние «active, nothing running», которое в живом TUI ждёт человека.
   Симулятор либо подстёгивает комментарием, либо фиксирует провал.
4. Никогда не трогает HOLD/удаления — сценарии их не используют.

Все действия симулятора пишутся в `runs/<run_id>/user_sim.jsonl` (что, когда,
какой статус у задачи был) — иначе в отчёте нельзя отличить «модель тупит» от
«симулятор вмешался».

---

## 7. Сценарии

Каждый сценарий — папка `scenarios/<id>/`:

```
scenarios/s3_compaction_coder/
├── seed/                  # сид-репозиторий: код + тесты, копируется в workspace
├── scenario.toml          # briefs, followups, clarification_answers, timeout
└── checks.py              # механическая проверка результата (вызывается харнессом)
```

Correctness-критерий всегда механический: сид содержит проверяемый контракт
(«функция X проходит тест Y»), харнесс **сам запускает `pytest` в workspace**
после прогона — вердикт не строится на словах агентов.

| ID | Сценарий | Что проверяет | Ключевые assertions |
|----|----------|---------------|---------------------|
| S1 | happy path: feature → кодер → тестер PASSED | базовая связка | `verification=PASSED`; ровно 1 testing-раунд; task COMPLETED; `checks.py` зелёный |
| S2a | seeded bug, tester FAILED → follow-up → фикс → PASSED | возврат на доработки через `send()` | ≥2 testing-раунда; финально PASSED; в транскрипте кодера виден follow-up одним prompt'ом |
| S2b | seeded bug, тестер вызывает `create_bug_report` | баг-репорт тестера в голове очереди | вторая задача kind=bug выполнена раньше прочих (`activated_at`); обе COMPLETED; баг-задача прошла testing |
| S3 | компакция у кодера: `max_context_tokens≈3000` + задача с многими `read_file` | mid-run pause → summary → continuation | событие «Context compacted» в `events`; `COMPACTION_SUMMARY_PREFIX` в `messages`; исходный prompt остался дословно; задача завершена, `checks.py` зелёный |
| S4 | компакция у тестера: тестер читает жирные выводы (`run_bash` с большим stdout) | mid-round compaction тестера | compaction-евент внутри testing-раунда; `tester_messages[-1]` содержит summary; verdict выставлен |
| S5 | clarification round-trip | `ask_clarifying_question` → ответ → голова очереди | статус clarification→queued→completed; ответ потреблён одним prompt'ом (нет дублей — см. `is_pending_retry`) |
| S6 | очередь: feature + bug одновременно | gate «никакой девелопмент, пока идёт testing»; приоритет багов | порядок активаций по `activated_at`; тестер не пересекается с кодером (по events/логу recorder) |
| S7 | fault injection через прокси (опция, §9) | ретраи и resume на настоящем HTTP | сетевые ретраи в `events`; `unfinished`-resume; финальный успех; число ретраев соответствует бюджету `NETWORK_RETRIES` |

Правила для всех сценариев:

- kinds только `feature`/`bug` (research не проходит тестера).
- Сид-репозиторий — маленький (пара файлов, 1–2 теста): цель бенчмарка —
  взаимодействия и тайминги, а не сложность задачи.
- Для S2a/S2b сид содержит заведомо сломанный код + тест, который его ловит.
- Для S4 тестер надо подвести к чтению жирного вывода: сид содержит файл с
  десятками тысяч строк, а brief прямо велит «проверь полный листинг X».

---

## 8. Матрица моделей и раннер

`matrix.toml`:

```toml
[providers.provider-a]
base_url = "https://api.provider-a.example/v1"
api_key  = "env:PROVIDER_A_KEY"        # поддержать подстановку из env хоста

[providers.provider-a.models.qwen3-coder-30b]
max_context_tokens = 131072
max_tokens = 16384

[providers.provider-b]
base_url = "http://localhost:8080/v1"
api_key  = ""

[[combo]]
name    = "strong-coder/cheap-tester"
coding     = "provider-a/qwen3-coder-30b"
tester     = "provider-b/qwen3-8b"
compaction = "provider-b/qwen3-8b"

[[combo]]
name    = "baseline-single-model"
coding     = "provider-a/qwen3-coder-30b"
tester     = "provider-a/qwen3-coder-30b"
compaction = "provider-a/qwen3-coder-30b"
```

Раннер: `bench run --scenario <id|all> --matrix matrix.toml --parallel N`.

- Полный прогон = N_scenarios × N_combos задач; `--parallel` ограничивает
  одновременные subprocess'ы (упирается в rate limits провайдера; для локальных
  шлюзов — в VRAM/слоты).
- Обязательные «угловые» комбинации: все роли на одной модели (базлайн),
  тестер слабее кодера (поймает ли он баг вообще), слабый компактизатор
  (деградация summary; failure компакции не должен ронять ран — это
  защищено в drcode, бенчмарк это подтверждает).
- Повторы: каждая пара прогоняется `--repeats K` (по умолчанию 3) —
  LLM недетерминированы, в отчёт идут медианы/разброс.
- Ретраи инфраструктурных сбоев прогона (упавший subprocess до старта
  приложения, недоступный провайдер на старте) — отдельно от содержательных
  провалов: сначала перепроверить конфиг, потом повторить прогон.

---

## 9. Прокси-режим и fault injection (опция)

Локальный HTTP-прокси (простой ASGI-сервер) между агентами и провайдером:

- base_url всех провайдеров в `models.json` указывает на прокси; прокси
  переписывает запросы на настоящий провайдер, логируя точные тайминги
  и полные тела (опционально).
- Fault injection: скриптованные 429, обрывы соединения, задержки — по счётчику
  запросов или по заголовку-маркеру. Достаточно для S7 без всяких правок
  drcode: сетевые ретраи, `unfinished`-resume и восстановление после outage
  проверяются на настоящем HTTP-слое.
- Прокси также заменяет recording-обёртку из §5, если она почему-то неудобна
  (например, при прогоне собранного бинаря) — но в основном режиме §5 проще.

---

## 10. Метрики и отчёт

### На прогон (runs/<run_id>/)

- `metrics.json` — всё, что извлечено (см. ниже).
- копия `<workspace>/.drcode/` (tasks, events, конфиги) — исходники для разбора.
- `model_log.jsonl` — записи recorder/прокси.
- `user_sim.jsonl` — вмешательства симулятора.
- stdout/stderr subprocess'а.

### Извлекаемые метрики

| Метрика | Источник |
|---|---|
| wall-time фаз: постановка → активация → конец девелопмента → конец testing → completed | `created_at`, `activated_at`, `completed_at`, timestamps `events` |
| точная латентность per model-request | recorder/прокси |
| токены per role (prompt/completion) | recorder; сверка с `usage` в сериализованных `ModelResponse` транскриптов |
| число model-запросов per agent per run | recorder |
| компакции: сколько, на какой фазе, когда | `events` («Context compacted», «pausing the run…», «Context fill N%») |
| экономия компакции | размер истории до/после (`n_history` в recorder; длина `messages` до/после summary-сообщения) |
| деградация после summary | успех `checks.py` + «compaction failed»-евенты + факт повторной компакции |
| корректность | `checks.py` (pytest в workspace) + `verification` |
| число testing-раундов, bug round-trip'ов | `tester_messages` (длина), задачи kind=bug и их `activated_at` |
| stability | «compaction failed», «tester failed», сетевые ретраи в `events`, `unfinished`-флаги |

### Отчёт

- `bench report runs/` → `report.md` (+ опционально HTML): таблица
  combo × scenario, строки — медиана времени, медиана токенов, вердикт,
  число компакций; разброс по повторам; ссылки на артефакты прогона.
- Каждый прогон должен быть воспроизводимо разобран: по `run_id` восстанавливаются
  конфиги, промпты, транскрипты и лог модели.

---

## 11. Структура репозитория drcode-bench

```
drcode-bench/
├── pyproject.toml            # python 3.12; deps: drcode (git), textual, rich
├── README.md
├── matrix.toml
├── bench/
│   ├── __init__.py
│   ├── __main__.py           # CLI: run / report
│   ├── run_one.py            # один прогон: env → DrCodeApp headless → settle
│   ├── env_setup.py          # tmp-дерево, генерация models.json/drcode.json/settings.json
│   ├── recording.py          # recording-обёртка build_model (§5)
│   ├── user_sim.py           # симулятор пользователя (§6)
│   ├── settle.py             # критерий завершения + диагностика провала
│   ├── metrics.py            # парсинг .drcode/tasks/*.json + model_log.jsonl
│   ├── proxy.py              # (опция) записывающий/fault-injecting прокси (§9)
│   └── report.py             # агрегация → markdown/HTML
├── scenarios/
│   ├── s1_happy/
│   ├── s2a_bug_followup/
│   ├── s2b_bug_report/
│   ├── s3_compaction_coder/
│   ├── s4_compaction_tester/
│   ├── s5_clarification/
│   ├── s6_queue_priority/
│   └── s7_fault_injection/
└── runs/                     # артефакты прогонов (gitignored)
```

---

## 12. Порядок работ

1. **Каркас + S1** на одной реальной модели: subprocess, headless `DrCodeApp`,
   settle-детект, извлечение метрик из task json, минимальный отчёт.
2. **Recording-обёртка** `build_model` (роль, латентность, usage) — без неё
   тайминги приблизительные, а usage компакций не виден вовсе.
3. **Симулятор пользователя** + интерактивные сценарии S2a/S2b/S5.
4. **Компактизационные сценарии S3/S4** на маленьком `max_context_tokens`.
5. **Матрица** комбинаций + параллельный раннер + повторы + полный отчёт.
6. **Прокси + fault injection (S7)** — опциональный последний этап.

---

## 13. Подводные камни (сводка)

- Порог компакции тестера считается от **кодинговой** модели — «маленький
  контекст» ставится кодеру даже в сценарии S4. В матрице учитывать; само это
  поведение — тоже объект проверки.
- `DRCODE_MODEL`/`DRCODE_BASE_URL` в окружении прогона перебивают per-role
  выборку для всех ролей — держать окружение чистым.
- Репозиторный `.env` drcode протекает через pydantic-settings при запуске из
  исходников — в bench-процессе окружение вычищать полностью (свежий subprocess).
- `git push` тестера: `task_finalization="testing"` или локальный bare-remote;
  никогда GitHub.
- `stall_nudge="never"` на время замеров (иначе watchdog оборвёт медленную
  модель); отдельный сценарий может намеренно его тестировать.
- Харнесс не «открывает» задачи в чате (`viewed_id` не трогать) — иначе
  рендер-оверхед попадает в замеры.
- kinds только `feature`/`bug`.
- LLM недетерминированы: повторы + медианы обязательны; единичный прогон — не
  результат, а точка данных.
- Стриминг модели в recording-обёртке должен оставаться стримингом — TUI и
  watchdog рассчитаны на дельты.
