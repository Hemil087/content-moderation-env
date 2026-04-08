---
title: Content Moderation Env Environment Server
emoji: 🛡️
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Content Moderation Environment

An OpenEnv environment where an LLM agent reviews social media posts, retrieves precedent cases, and makes moderation decisions under platform policy constraints.

**HF Space:** [https://hemil087-content-moderation-env.hf.space](https://hemil087-content-moderation-env.hf.space)

---

## Overview

Content moderation is one of the hardest real-world challenges facing social media platforms. Moderators must balance free expression against harmful content, handle ambiguous edge cases, and maintain consistency across millions of decisions. This environment simulates that task for LLM agents.

The agent receives a reported post with rich metadata (author history, engagement metrics, reporter credibility, automated classifier scores), the platform's community policy, and access to a database of 20 precedent cases. It must decide whether to remove, allow, label, escalate, or (for appeals) overturn/uphold prior decisions.

---

## Tasks

| Task | Difficulty | Episodes | Description |
|------|-----------|----------|-------------|
| `easy` | Low | 5 | Clear policy violations (hate speech, dehumanization) or clearly allowed content (news, political opinion). Minimal ambiguity. |
| `medium` | Medium | 5 | Ambiguous content requiring careful precedent analysis — coded language, edgy satire, graphic news with temporal context. |
| `hard` | High | 7 | Appeal cases with conflicting precedents, coded dog-whistle language, compelling but wrong arguments, legal threats. Requires nuanced policy interpretation. |

Episodes are selected via `episode_index` in reset options (default: 0). Each episode has deterministic grading — no randomness in scoring.

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

# Make a final decision with reason (citing policy clause improves score)
action = ContentModerationAction(
    action_type="remove_content",
    reason="Dehumanizing language targeting protected ethnic group"
)
```

---

## Observation Space

After each step, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `post_content` | `str` | The post or comment being reviewed |
| `post_metadata` | `dict` | Rich metadata: report count, author history, engagement, auto-flag score, reporter credibility, content type, temporal context |
| `policy_summary` | `str` | Platform community policy (always visible) |
| `precedents` | `list` | Retrieved precedent cases (populated after `retrieve_precedents`) |
| `similar_post_count` | `int` | Number of similar posts on platform |
| `confidence_guidance` | `str` | Guidance on certainty required |
| `actions_taken` | `list` | History of actions taken this episode |
| `step_count` | `int` | Current step number |
| `message` | `str` | Feedback from environment (including precedent conflict warnings) |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward for the last action |

### Post Metadata Fields

Each post includes realistic Trust & Safety queue metadata:

| Field | Description |
|-------|-------------|
| `author_account_age_days` | Account age (15 = brand new, 3650 = 10yr veteran) |
| `author_prior_violations` | Previous policy violations |
| `author_verified` | Verified account status |
| `author_follower_count` | Audience reach |
| `content_type` | text_post, comment, shared_post, group_post, appeal, direct_message_report |
| `has_media` | Whether post includes image/video |
| `engagement_count` | Total likes + shares |
| `reporter_accuracy_rate` | Reporter's historical accuracy (trusted flagger vs mass-reporter) |
| `report_category` | What the reporter flagged it as |
| `auto_flag_score` | Automated classifier confidence (0.0–1.0) |
| `current_events_context` | Active platform protocols (elections, disasters) — present on some episodes |
| `original_moderator_reasoning` | How original decision was made — present on appeal episodes |

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Retrieved relevant precedent | +0.15 |
| Retrieved irrelevant precedent | -0.10 |
| Correct final decision | +0.50 |
| Wrong final decision | -0.30 |
| Near-miss decision (adjacent action) | -0.15 |
| Cited correct policy clause in reason | +0.10 |
| No reason provided with final action | -0.05 |
| Relevant precedent bonus (grader) | +0.15 |
| Same action called twice (loop) | -0.20 |
| Hit step limit without deciding | -0.30 |

Final episode score is clamped to 0.0–1.0.

**Maximum achievable score:** 0.90 (retrieve relevant + correct decision + relevant precedent bonus + policy clause citation)

**Adjacent action pairs** that receive reduced penalty (-0.15 instead of -0.30):
- `remove_content` ↔ `add_warning_label`
- `allow_content` ↔ `add_warning_label`
- `overturn_removal` ↔ `allow_content`
- `uphold_removal` ↔ `remove_content`

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
        reason="Dehumanizing language targeting protected group"
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
export HF_TOKEN=your_hf_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Baseline Scores

| Task | Score |
|------|-------|
| Easy | 0.90 |
| Medium | 0.90 |
| Hard | 0.90 |
| Average | 0.90 |

Model: `llama-3.3-70b-versatile` via Groq

---

## Local Development

```bash
# Clone and install
git clone https://github.com/Hemil087/content-moderation-env.git
cd content-moderation-env
pip install -e .

# Start server
uvicorn content_moderation_env.server.app:app --reload

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"options": {"task_id": "easy", "episode_index": 0}}'
```

---

## Project Structure

```
content-moderation-env/
├── inference.py                          # LLM agent script (repo root)
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

## Environment Design

### Precedent Database
20 fixed precedent cases covering hate speech, political criticism, satire, news reporting, graphic content, and more. Precedents are retrieved via keyword search — deterministic, no external API calls.

### Grading
All grading is deterministic (pure Python, no LLM calls). The grader checks:
1. Was the final action correct?
2. Did the agent retrieve relevant precedents?
3. Did the agent cite the correct policy clause in their reason?
4. Did the agent provide a reason at all?

### Precedent Conflict Detection
When retrieved precedents have conflicting decisions (e.g., one says "allowed", another says "removed"), the environment flags this in the message, forcing the agent to reason about which precedent applies.

---

## License

BSD 3-Clause License