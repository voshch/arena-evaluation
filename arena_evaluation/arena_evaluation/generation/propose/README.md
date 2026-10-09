# propose

The one model call path, and the zone roles no substring rule reaches. Part of [generation](../README.md).

| File | Purpose |
|---|---|
| `client.py` | `ask`: Gemini through `task_generator.utils.gpt`, temperature 0, 120 s timeout, one retry; default model `$ARENA_PROMPT_LLM` |
| `zone_roles.py` | `classify_tail` sends only the unclassified names with their size and contents; `merge` writes `../scenario/zone_roles.yaml` with the model id, `prompt_sha` and the model's reason |

`classify` asks about the tail only: the rules cover the common names for free and deterministically, and `unknown` is a permitted answer that routes a placeholder name to a human.
