# Дизайн: встроенный MCP-сервер в живой TUI (Streamable HTTP, localhost)

Статус: черновик на утверждение (research/design — без реализации).
Задача: дать внешним ИИ-агентам управлять работой drcode через MCP — ставить
задачи, следить за статусом, отвечать на clarifications, двигать очередь.

## 0. Ключевое архитектурное решение

**Сервер встраивается в TUI-процесс** (не headless): мутации жизненного цикла
должны проходить через живой pipeline — единый гейт `_advance()`
(`core/pipeline.py`) живёт в экземпляре приложения, и только он честно
обрабатывает параллельные слоты, TESTING-бэклог, dependency-гейты и
`promote_next`. Headless CLI (`cli.py`) работает с TaskManager в отрыве от
pipeline и потому годится только для импорта задач, не для управления.

Модули сервера живут в **`src/drcode/mcp/`** (НЕ в `core/` — туда запрещён
любой UI/серверный код, `tests/test_core_purity.py`; импорт из `core` наружу
— разрешён и нужен). Зависимость: официальная **`mcp` SDK**
(modelcontextprotocol/python-sdk) — pydantic v2-native, anyio-based,
Streamable HTTP из коробки; тянет starlette + uvicorn + sse-starlette +
httpx (httpx уже core-dep, pydantic уже есть). Standalone `fastmcp` отклонён:
тот же SDK под капотом, но больше транзитивных зависимостей и резче churn —
для фиксированного самописного tool-набора его надстройки (middleware,
proxying, composition) не нужны. Ручная реализация JSON-RPC поверх aiohttp
отклонена: protocol churn (версии спецификации, сессии, framing) дороже
зависимости. В `drcode.spec` добавить `mcp`, `uvicorn`, `starlette`,
`sse-starlette` в hiddenimports/collect (см. готчу про lazy-imported
textual.submodules).

**Тредовая модель.** uvicorn-сервер крутится в daemon-потоке со своим
event loop (паттерн notifications.py — «daemon thread, failures swallowed»).
Обработчики инструментов:

- *чистые мутации TaskManager* — синхронные вызовы `manager.*` напрямую
  (TaskManager потокотолерантен: `fire_status_change` уже бьёт из worker-threads,
  stats берёт lock);
- *шаг pipeline* — через `app.call_from_thread(app._advance)` (готовый паттерн:
  `tui/merge/screen.py`, `voice_input.py`, `app.py:1277`).

Исключения никогда не рвут TUI: любой сбой обработчика → MCP `isError`
результат с человекочитаемым текстом (та же философия, что `tool_runtime`:
fixable → сообщение, остальное — logged, never fatal).

## 1. Транспорт

**Streamable HTTP на 127.0.0.1** (MCP spec 2025-03-26+; HTTP+SSE устарел —
не используем). Stateful-режим SDK (дефолт): `Mcp-Session-Id` управляется SDK,
GET-SSE-стрим на том же эндпоинте доставляет серверные нотификации (см. §5).

- Эндпоинт: `POST/GET/DELETE http://127.0.0.1:<port>/mcp` + `GET /healthz`.
- Порт: **эфемерный свободный** (bind `127.0.0.1:0` при старте, когда
  включено), публикуется в **`.drcode/mcp.json`** (см. §2) — клиенты и shim
  не должны угадывать порт, а жёсткий дефолтный порт рано или поздно
  столкнётся с чужим сервисом.

**fs-based транспорт — отклонён** (совпадает с выводом исследования): нет в
спецификации MCP, популярные клиенты не поддерживают, выгода против
localhost-HTTP отсутствует.

**stdio-шим — да, включаем в скоуп**: точка входа `drcode-mcp`
(=`python -m drcode.mcp_proxy`, отдельный console script в pyproject).
Внешний клиент, умеющий только stdio, спавнит shim; тот поднимает
`mcp.server.stdio` с одной стороны и `mcp.client.streamable_http` с другой
и транслирует JSON-RPC между ними. Эндпоинт+токен shim берёт из
`<cwd>/.drcode/mcp.json` (клиенты спавнят MCP-серверы в корне проекта —
cwd совпадает с корнем TUI). Эндпоинт недоступен → shim выдаёт внятную
ошибку в stderr («drcode TUI не запущен или MCP выключен — включите в
Settings → MCP server») и завершается с кодом 1; авто-старт TUI из shim —
никогда (мутации обязаны идти через живой pipeline). Опции `--port/--token`
— для нестандартных сценариев, файл остаётся основным источником.

## 2. Аутентификация и включение

- **Включение**: project setting `mcp_enabled: bool = False` в
  `.drcode/settings.json` (`settings.py`: поле `ProjectSettings` + пикер в
  Settings-палетре по образцу `subagents_coder`). Выключено → сокет вообще
  не биндится. Тумблер в рантайме: вкл/выкл поднимает/гасит сервер сразу
  (hot start/stop), сервис-ивент в чат.
- **Токен**: 32 байта hex (`secrets.token_hex(32)`), генерируется при первом
  включении, хранится в **`.drcode/mcp.json`** (не settings.json: это
  секрет, а settings.json пользователь может коммитить/читать глазами):
  `{"port": 51234, "token": "...", "pid": <tui-pid>, "started_at": "..."}`.
  Файл писать с правами 0600; `.drcode/` уже в gitignore. Потеря файла →
  перегенерация токена при следующем старте сервера (подключившиеся клиенты
  переконфигурируют shim — приемлемо, событие редкое).
- **Проверка**: `Authorization: Bearer <token>` обязателен на `/mcp` (POST,
  GET-SSE, DELETE); невалидный → 401 без деталей. Дополнительно к токену —
  стандартная для HTTP-транспортов защита от DNS-rebinding: валидация
  `Origin`/`Host` (принимать только отсутствующий Origin и localhost) —
  в SDK это middleware над ASGI-аппой, оборачиваем наш `streamable_http_app()`.
- **`/healthz`** — без токена, ответ `{"ok": true}` и больше ничего
  (используется shim'ом и второй инстансом для пробы; утечки данных нет).
- CORS — не настраиваем (никаких браузерных клиентов не предполагается).

## 3. Один сервер = один проект

Корень = cwd TUI. Никакого мультипроекта в одном сервере (иначе внешние
агенты смогут мутировать чужие проекты — разрыв модели безопасности).

**Два окна одного проекта**: desktop-шелл блокирует два таба одного проекта,
но два голых `uv run drcode` в одном cwd возможны — два TaskManager над
одними файлами и так опасны. Правило старта сервера:

1. `mcp_enabled` и `.drcode/mcp.json` существует и `/healthz` отвечает, а
   `pid` в файле жив → **не стартуем свой сервер**, пишем сервис-ивент
   «MCP: endpoint уже обслуживается другим экземпляром (pid N)» — наружу
   торчит ровно один сервер, второй TUI работает молча без него.
2. Файл есть, но проба не отвечает (мертвый TUI) или `pid` мёртв → файл
   затирается новым, сервер стартует.
3. Файла нет → создаём, стартуем.

При завершении TUI — graceful shutdown (§6) и удаление `mcp.json`, если
`pid` в нём наш (не наш — не трогаем: его пишет кто-то другой).

## 4. Мутации и статусы задач

Принцип: **MCP не может то, чего не может человек в TUI**. Матрица
допустимости повторяет TUI-поведение (`actions_lifecycle.py`), а не выдумывает
новое:

| Инструмент | queued/draft/hold | clarification | ACTIVE/TESTING | completed |
|---|---|---|---|---|
| `send_input` | да | да (= ответ) | **да** — как комментарий в TUI: прерывает run, текст идёт в pending_input, run перезапускается | перезакидывает в очередь (снимая hold-флаги) |
| `answer_clarification` | — | да | — | — |
| `edit_task` | по `editable_condition` (причина в ошибке) | нет | нет | нет |
| `set_dependencies`, `set_kind` | да | нет | **ошибка**: «задача ACTIVE/TESTING — отмените (cancel) или дождитесь» | нет |
| `set_hold(on)` | да | нет | ошибка | нет |
| `move_in_queue` | да | — | — | — |
| `cancel` | да | да | **да** (power-action как Ctrl+X: abort + COMPLETED, мимо тестера — явно документировать в описании инструмента) | нет |
| `delete` | только hold/draft (контракт `manager.delete`) | нет | нет | нет |

Попытка запрещённой мутации — **не исключение, а нормальный MCP `isError`**
с текущим статусом и советом. `trigger_advance` — явный вызов `_advance()`
через `call_from_thread` (кнопка «nudge» для внешнего агента; обычно не
нужен — мутации сами планируют advance, см. §6).

**Не публикуются** (подтверждено): `set_verification`/`finalize_verified`
(COMPLETED легитимно только от тестера или человека),
`requeue_failed_fix`/`requeue_review_rejection` (внутренности tester/hook),
`complete_without_verification`, любая запись транскриптов/скелета,
конфигурация моделей (`providers`/каталог), `purge_expired_completed`,
`link_goal`/`unlink_goal` (v1: `goal_id` задаётся при `create_task`;
долинки — позже, если попросят).

## 5. События

Слушатели TaskManager (`add_status_listener` / `add_removal_listener` /
`add_completion_listener` / `add_creation_listener`) уже есть; `fire_status_change`
— единая воронка всех переходов. Мост:

1. TUI регистрирует в менеджере один тонкий listener, который кладёт
   событие в потокобезопасную очередь (`loop.call_soon_threadsafe` в loop
   MCP-потока).
2. MCP-сервер транслирует их в **MCP resources**: `drcode://queue`,
   `drcode://tasks/<id>`, `drcode://goals/<id>`, `drcode://stats`. Клиент
   подписывается штатным `resources/subscribe`, сервер шлёт штатную
   `notifications/resources/updated` — никаких кастомных нотификаций,
   которые клиенты могут не понимать. Коалесцирование: всплеск событий
   за ~250 мс склеивается в одну нотификацию на URI (без этого burst из
   50 переходов зальёт клиента).
3. Агент без подписок не теряет ничего: `list_tasks`/`get_task` — polling.

Дедуп/маршрутизация: нотификация только по URI затронутых ресурсов;
origin-фильтр не нужен (внешнему агенту интересны все переходы).

## 6. Жизненный цикл внутри Textual

- **Старт**: `app.on_mount` при `settings.mcp_enabled` → выбор свободного
  порта → запись `mcp.json` → `uvicorn.Server` в daemon-потоке
  (`uvicorn.Server(config).run()` в `threading.Thread`; ASGI-аппа — SDK'шный
  streamable HTTP, обёрнутый auth/origin-middleware).
- **Остановка**: `on_unmount` → `server.should_exit = True` + `join(timeout=5)`
  (паттерн voice/notifications: daemon-поток не мешает выходу, но корректный
  shutdown снимает сокет до завершения) → удаление своего `mcp.json`.
- **Конфликт портов**: не возникает по построению (эфемерный порт); единственный
  сценарий — гонка двух TUI между пробой и биндом — закрывается шагами 1–3 §3
  (второй инстанс перепроверяет healthz до записи файла).
- **UI-видимость**: индикатор в статус-баре/header («MCP :51234») и
  сервис-ивент при старте/остановке/отказе (например, «MCP: уже обслуживается
  pid 1234»). Токен в UI не показывается — только путь к `mcp.json`.
- **Re-advance после мутаций**: каждый успешный мутирующий вызов, после
  которого TUI нажал бы Enter (create/send/answer/cancel/set_hold(off)/
  move_in_queue/delete), планирует `call_from_thread(app._advance)` — ровно
  как `actions_lifecycle.py`. Гонки невозможны: всё уходит в один event loop
  приложения.

## 7. Список инструментов (финальный)

**Чтение** (все — дешёвые метаданные, транскрипты никогда не читаются;
`skeleton` — только в `get_task`):

| Инструмент | Основа | Возвращает |
|---|---|---|
| `list_tasks(status?, kind?, limit?)` | `manager.all_tasks` | компактные строки: id, title, kind, status, verification, hold, unfinished, needs_human, github_issue, goal_id, pending_input count, has_clarification |
| `get_task(task_id)` | `manager.get` | полные метаданные + skeleton (промпты юзера + финальный ответ) + events (последние N) + clarification-пакет + blocking_dependencies |
| `search_tasks(keywords)` | `task_search` | до 50 строк с id/статусом/сниппетом (AND→OR семантика модуля) |
| `get_queue()` | `manager.queued` + `blocking_dependencies` | порядок очереди + чем заблокирована каждая задача |
| `list_goals()` / `get_goal(id)` | `GoalStore` + `manager.goal_progress/goal_complete` | цели с прогрессом done/total |
| `get_stats(days?)` | `StatsStore` | дневные агрегаты (created/completed/dev_seconds/…) |
| `get_usage(days?)` | `usage_store` | пер-дневные счётчики по `<role>/<model>` |
| `check_file_overlaps(text, depends_on?)` | `task_overlap.find_file_overlaps` | предфлайт-пересечение по файлам (create_task прогоняет его и так) |

**Мутации** (см. матрицу §4): `create_task(title?, text, kind,
depends_on?, goal_id?, hold?)`, `edit_task(task_id, text, kind?)`,
`send_input(task_id, text)`, `answer_clarification(task_id, answers[]):
count валидируется против `clarification.questions`, ответы склеиваются
«по строке на вопрос» — ровно то, что TUI делает по Enter; вопрос с
вариантами отвечает текстом варианта), `set_dependencies`,
`set_kind`, `set_hold`, `move_in_queue`, `cancel`, `delete`,
`trigger_advance()`.

**Ресурсы**: `drcode://queue`, `drcode://tasks/{id}`, `drcode://goals/{id}`,
`drcode://stats` (подписки + `resources/updated`, §5).

## 8. Тестирование и валидация

- `tests/test_mcp_server.py`: in-process — SDK-клиент через ASGI-транспорт
  (без реальных сокетов) для tool-контрактов; auth (401 без/с чужим токеном),
  origin-reject, gating-матрица §4, коалесцирование нотификаций.
- Реальный сокет — один smoke-тест: поднять сервер на `127.0.0.1:0`,
  подключиться streamable-клиентом, вызвать `list_tasks` (гейт на CI нет —
  локальный, стабильный).
- Второй-инстанс логика: unit на probe+pid (файл жив/мёртв).
- Ruff/pyright как обычно; `-n 0` для фокуса, `-n 4` для регресса.

## 9. Открытые точки (за пределами v1)

- Долинки целей (`link_goal`/`unlink_goal`) отдельными инструментами.
- Стриминг прогресса активного рана наружу (сейчас — только статусы).
- Публичный `ask drcode` для research-ответов архива вне задач.
