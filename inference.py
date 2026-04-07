"""
Baseline inference script for the Content Moderation Environment.

Runs an LLM agent against all 3 tasks (easy, medium, hard) and produces
reproducible baseline scores.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL    - The API endpoint for the LLM
    MODEL_NAME      - The model identifier to use for inference
    HF_TOKEN        - Your Hugging Face API key
    OPENAI_API_KEY  - OpenAI API key (used by OpenAI client)

Logs:
    Emits structured JSON to stdout in [START], [STEP], [END] format.
    Any deviation from this format will break evaluation scoring.
"""

import json
import os
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get(
    "ENV_URL",
    "https://hemil087-content-moderation-env.hf.space"
)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 8

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert content moderator for a social media platform.

Your job is to review posts and make fair, consistent moderation decisions
based on the platform's community policy.

## AVAILABLE ACTIONS

You must respond with a JSON object choosing ONE of these actions:

1. retrieve_precedents — Search for similar past cases to guide your decision
   {"action_type": "retrieve_precedents", "query": "your search query here"}

2. remove_content — Remove the post for policy violation
   {"action_type": "remove_content", "reason": "explanation here"}

3. allow_content — Approve the post as policy-compliant
   {"action_type": "allow_content", "reason": "explanation here"}

4. add_warning_label — Keep post but add a content warning
   {"action_type": "add_warning_label", "reason": "explanation here"}

5. escalate — Send to senior human moderator
   {"action_type": "escalate", "reason": "explanation here"}

6. overturn_removal — For appeals: reverse the prior removal decision
   {"action_type": "overturn_removal", "reason": "explanation here"}

7. uphold_removal — For appeals: confirm the prior removal was correct
   {"action_type": "uphold_removal", "reason": "explanation here"}

## MANDATORY STRATEGY

You MUST follow this exact two-step process:

STEP 1 (ALWAYS): Call retrieve_precedents first with a query describing the key issue.
   - For hate speech cases, query: "dehumanizing [target group]"
   - For political content, query: "political criticism opinion"
   - For satire/comedy, query: "satire comedy offensive"
   - For graphic content, query: "graphic disturbing news"
   - For appeals, query: "appeal [original removal reason]"

STEP 2: After reviewing precedents, make your final decision based on:
   - The platform policy
   - The precedent cases you retrieved
   - The specific content and context

## DECISION GUIDELINES

- Content that dehumanizes protected groups (race, religion, gender, sexuality, disability) → remove_content
- Political criticism targeting policies or public figures' actions → allow_content
- Factual news reporting, even on sensitive topics → allow_content
- Historical quotes with educational context → allow_content
- Graphic/disturbing content with news value → add_warning_label
- Edgy comedy or satire that is ambiguous → add_warning_label
- User threatens legal action or to leave platform → escalate
- Appeals where original removal was wrong (e.g., news/satire removed) → overturn_removal
- Appeals where original removal was justified → uphold_removal

## CRITICAL RULES
- ALWAYS retrieve precedents first. Never skip this step.
- ONLY escalate when the user explicitly threatens legal action or to leave the platform.
- When unsure between remove and warning label, prefer add_warning_label for comedy/satire.
- Never call the same action twice — it will be penalized.

## RESPONSE FORMAT

Always respond with ONLY a valid JSON object. No explanation outside the JSON.
Example: {"action_type": "retrieve_precedents", "query": "hate speech dehumanizing ethnicity"}
""".strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def build_user_message(observation: dict) -> str:
    """Build the user message from the current observation."""
    parts = []

    parts.append(f"POST TO REVIEW:\n{observation.get('post_content', '')}")

    metadata = observation.get("post_metadata", {})
    if metadata:
        parts.append(f"POST METADATA:\n{json.dumps(metadata, indent=2)}")

    policy = observation.get("policy_summary", "")
    if policy:
        parts.append(f"PLATFORM POLICY:\n{policy}")

    precedents = observation.get("precedents", [])
    if precedents:
        prec_text = "\n".join([
            f"[{p['case_id']}] Decision: {p['decision']} | Reason: {p['reason']}\nContent: {p['content']}"
            for p in precedents
        ])
        parts.append(f"RETRIEVED PRECEDENTS:\n{prec_text}")

    actions_taken = observation.get("actions_taken", [])
    if actions_taken:
        parts.append(f"ACTIONS TAKEN SO FAR: {', '.join(actions_taken)}")

    message = observation.get("message", "")
    if message:
        parts.append(f"ENVIRONMENT FEEDBACK: {message}")

    return "\n\n".join(parts)


def get_llm_action(conversation_history: list) -> dict:
    """Call LLM and parse action from response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,
        temperature=0.0,  # deterministic for reproducibility
        max_tokens=256,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        # Fallback — escalate if we can't parse
        action = {"action_type": "escalate", "reason": "Unable to parse LLM response"}

    return action, content


def run_episode(env_client, task_id: str) -> float:
    """
    Run one full episode for the given task_id.
    Returns the final score (0.0–1.0).
    Emits structured stdout logs.
    """
    print(json.dumps({"type": "START", "task_id": task_id}))

    # Reset environment
    result = env_client.reset(options={"task_id": task_id})
    obs = result.observation

    total_reward = 0.0
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_actions = {
        "remove_content",
        "allow_content",
        "add_warning_label",
        "escalate",
        "overturn_removal",
        "uphold_removal",
    }

    for step in range(MAX_STEPS):
        if obs.done:
            break

        # Build user message from current observation
        user_message = build_user_message(obs.__dict__)
        conversation_history.append({"role": "user", "content": user_message})

        # Get LLM action
        action_dict, raw_response = get_llm_action(conversation_history)

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": raw_response})

        # Build action object
        from content_moderation_env.models import ContentModerationAction
        action = ContentModerationAction(
            action_type=action_dict.get("action_type", "escalate"),
            reason=action_dict.get("reason"),
            query=action_dict.get("query"),
        )

        # Step environment
        result = env_client.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        total_reward = obs.reward if done else total_reward + reward

        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": step + 1,
            "action": action_dict,
            "reward": reward,
            "done": done,
            "message": obs.message,
        }))

        if done:
            total_reward = obs.reward  # final graded score
            break

    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "total_reward": total_reward,
    }))

    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from content_moderation_env.client import ContentModerationEnv

    print(json.dumps({
        "type": "INFO",
        "model": MODEL_NAME,
        "env_url": ENV_URL,
        "tasks": TASKS,
    }))

    scores = {}

    with ContentModerationEnv(base_url=ENV_URL).sync() as env:
        for task_id in TASKS:
            score = run_episode(env, task_id)
            scores[task_id] = score

    print(json.dumps({
        "type": "SUMMARY",
        "scores": scores,
        "average": round(sum(scores.values()) / len(scores), 4),
    }))


if __name__ == "__main__":
    main()