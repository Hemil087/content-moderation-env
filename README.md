# Content Moderation Environment

An OpenEnv environment where an LLM agent reviews social media posts, retrieves precedent cases, and makes moderation decisions under platform policy constraints.

**HF Space:** [https://hemil087-content-moderation-env.hf.space](https://hemil087-content-moderation-env.hf.space)

---

## Overview

Content moderation is one of the hardest real-world challenges facing social media platforms. Moderators must balance free expression against harmful content, handle ambiguous edge cases, and maintain consistency across millions of decisions. This environment simulates that task for LLM agents.

The agent receives a reported post, the platform's community policy, and access to a database of 20 precedent cases. It must decide whether to remove, allow, label, escalate, or (for appeals) overturn/uphold prior decisions.

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | Low | Clear policy violations or clearly allowed content. Minimal ambiguity. |
| `medium` | Medium | Ambiguous content (satire, graphic news, coded language) requiring precedent lookup. |
| `hard` | High | Appeal cases with conflicting precedents, or content where the user threatens legal action. |

Each task tier contains 5 fixed episodes. Episodes are selected deterministically using seeded randomness for reproducibility.

---

## Action Space

| Action | Description | When to Use |
|--------|-------------|-------------|
| `retrieve_precedents` | Search the precedent database for similar past cases | Before making any final decision |
| `remove_content` | Remove the post for policy violation | Dehumanizing content targeting protected groups |
| `allow_content` | Approve the post as policy-compliant | Political criticism, news reporting, historical context |
| `add_warning_label` | Keep post but attach a warning label | Graphic content, ambiguous satire, potentially misleading |
| `escalate` | Send to senior human moderator | User threatens legal action or to leave platform |
| `overturn_removal` | Reverse a prior removal (appeals only) | Original removal was unjustified (e.g., news/satire removed) |
| `uphold_removal` | Confirm a prior removal (appeals only) | Original removal was correct |

```python
from content_moderation_env.models import ContentModerationAction

# Retrieve precedents
action = ContentModerationAction(
    action_type="retrieve_precedents",
    query="hate speech dehumanizing ethnicity"
)

# Make a final decision
action = ContentModerationAction(
    action_type="remove_content",
    reason="Dehumanizing language targeting ethnic group"
)
```

---

## Observation Space

After each step, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `post_content` | str | The post or comment being reviewed |
| `post_metadata` | dict | Platform, report count, timestamp, is_appeal flag |
| `policy_summary` | str | Platform community policy (always visible) |
| `precedents` | list | Retrieved precedent cases (populated after `retrieve_precedents`) |
| `actions_taken` | list | History of actions taken this episode |
| `step_count` | int | Current step number |
| `message` | str | Feedback from environment |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward for the last action |

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Retrieved relevant precedent | +0.15 |
| Retrieved irrelevant precedent | -0.10 |
| Correct final decision | +0.50 |
| Wrong final decision | -0.30 |
| Relevant precedent bonus (grader) | +0.15 |
| Same action called twice (loop) | -0.20 |
| Hit step limit without deciding | -0.30 |

Final episode score is clamped to **0.0–1.0**. Maximum achievable score per episode: **0.8**.

---

## Quick Start

### Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/Hemil087/content_moderation_env
```

### Use (Sync)

```python
from content_moderation_env.client import ContentModerationEnv
from content_moderation_env.models import ContentModerationAction

with ContentModerationEnv(base_url="https://hemil087-content-moderation-env.hf.space").sync() as env:
    result = env.reset(options={"task_id": "easy"})
    print(result.observation.post_content)

    # Retrieve precedents
    action = ContentModerationAction(
        action_type="retrieve_precedents",
        query="hate speech dehumanizing"
    )
    result = env.step(action)
    print(result.observation.precedents)

    # Make final decision
    action = ContentModerationAction(
        action_type="remove_content",
        reason="Dehumanizing language"
    )
    result = env.step(action)
    print(f"Score: {result.reward}, Done: {result.done}")
```

### Use (Async)

```python
import asyncio
from content_moderation_env.client import ContentModerationEnv
from content_moderation_env.models import ContentModerationAction

async def main():
    async with ContentModerationEnv(base_url="https://hemil087-content-moderation-env.hf.space") as env:
        result = await env.reset(options={"task_id": "medium"})
        print(result.observation.post_content)

asyncio.run(main())
```

---

## Running Inference

The baseline agent uses an LLM via the OpenAI-compatible API:

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export OPENAI_API_KEY=your_key_here
python inference.py
```

On Windows:
```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set OPENAI_API_KEY=your_key_here
python inference.py
```

### Baseline Scores

| Task | Score |
|------|-------|
| Easy | 0.80 |
| Medium | 0.80 |
| Hard | 0.80 |
| **Average** | **0.80** |

Model: `llama-3.3-70b-versatile` via Groq

---

## Local Development

```bash
# Clone and install
git clone https://github.com/Hemil087/content-moderation-env.git
cd content-moderation-env
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Start server
uvicorn content_moderation_env.server.app:app --reload

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset

# Run local test (no server needed)
python test_local.py
```

---

## Project Structure

```
content-moderation-env/
├── inference.py                          # LLM agent script (repo root)
├── test_local.py                         # Local environment test
├── Dockerfile                            # Container build
├── requirements.txt                      # Dependencies
├── content_moderation_env/
│   ├── __init__.py
│   ├── models.py                         # Action + Observation definitions
│   ├── client.py                         # WebSocket client
│   ├── openenv.yaml                      # Environment manifest
│   ├── pyproject.toml                    # Package config
│   └── server/
│       ├── app.py                        # FastAPI application
│       └── content_moderation_env_environment.py  # Core environment logic
```

---

## License

BSD 3-Clause License