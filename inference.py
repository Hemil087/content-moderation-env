"""
Baseline inference script for the Content Moderation Environment.

Runs an LLM agent against all 3 tasks (easy, medium, hard) and produces
reproducible baseline scores.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL    - The API endpoint for the LLM
    MODEL_NAME      - The model identifier to use for inference
    HF_TOKEN        - Hugging Face API key (primary)
    OPENAI_API_KEY  - OpenAI API key (fallback)
"""

import json
import os
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv(
    "ENV_URL",
    "https://hemil087-content-moderation-env.hf.space",
)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 8
ENV_NAME = "content_moderation_env"

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ---------------------------------------------------------------------------
# Structured logging helpers — exact format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
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
- ALWAYS include a "reason" field citing the specific policy clause that applies.

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


def get_llm_action(conversation_history: list) -> tuple:
    """Call LLM and parse action from response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,
        temperature=0.0,
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
        action = {"action_type": "escalate", "reason": "Unable to parse LLM response"}

    return action, content


def run_episode(env_client, task_id: str) -> float:
    """
    Run one full episode for the given task_id.
    Returns the final score (0.0-1.0).
    Emits structured stdout logs in [START]/[STEP]/[END] format.
    """
    rewards = []
    steps_taken = 0
    score = 0.0

    log_start(task_id, ENV_NAME, MODEL_NAME)

    try:
        # Reset environment
        result = env_client.reset(options={"task_id": task_id})
        obs = result.observation

        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(MAX_STEPS):
            if obs.done:
                break

            # Build user message from current observation
            user_message = build_user_message(obs.__dict__)
            conversation_history.append({"role": "user", "content": user_message})

            # Get LLM action
            action_dict, raw_response = get_llm_action(conversation_history)
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

            steps_taken = step + 1
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action_dict.get("action_type", "unknown"),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                score = obs.reward  # final graded score
                break

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", file=sys.stderr)

    finally:
        success = score > 0.0
        log_end(success, steps_taken, score, rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from content_moderation_env.client import ContentModerationEnv

    with ContentModerationEnv(base_url=ENV_URL).sync() as env:
        for task_id in TASKS:
            run_episode(env, task_id)


if __name__ == "__main__":
    main()