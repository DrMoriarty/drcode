# drcode

Console AI coding agent (pydantic-ai + textual), inspired by opencode.
The center is not a chat session, but a **programming task** (feature request or bug report):

- one task is *active* at a time — work on it looks like a dialog that starts with the task description;
- while the agent works, you can create new tasks and put them in a **queue**;
- when the active task is marked done, the next queued task starts **automatically**;
- you can browse completed tasks and write into one — it re-enters the queue (or becomes active immediately if nothing is running).

## Install & run

```bash
uv sync
uv run drcode
```

## Configuration

Point it to any OpenAI-compatible API via env vars (or a `.env` file in the project root):

```bash
DRCODE_MODEL=gpt-4o
DRCODE_BASE_URL=https://api.openai.com/v1   # or your own endpoint
DRCODE_API_KEY=sk-...
```

## Keys

| Key     | Action                                              |
| ------- | --------------------------------------------------- |
| `ctrl+n` | New task (title + description)                      |
| `ctrl+d` | Mark viewed task done → next queued starts          |
| `ctrl+o` | Focus task sidebar (↑/↓ + Enter to switch tasks)    |
| `ctrl+i` | Focus chat input                                    |
| `escape` | Abort the running agent                             |
| `ctrl+q` | Quit                                                |

## Storage

Tasks and full transcripts (including tool calls) are stored as JSON under
`.drcode/tasks/` in the project root.
