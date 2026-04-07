"""
Quick local test — runs the environment directly (no server needed).
"""

import json
from content_moderation_env.models import ContentModerationAction
from content_moderation_env.server.content_moderation_env_environment import (
    ContentModerationEnvironment,
)


def test_episode(env, task_id):
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")

    obs = env.reset(options={"task_id": task_id})
    print(f"Post: {obs.post_content[:80]}...")
    print(f"Reports: {obs.post_metadata.get('report_count')}")
    print(f"Is appeal: {obs.post_metadata.get('is_appeal', False)}")

    # Step 1 — retrieve precedents
    action1 = ContentModerationAction(
        action_type="retrieve_precedents",
        query="hate speech dehumanizing immigration"
    )
    obs = env.step(action1)
    print(f"\nStep 1 — retrieve_precedents")
    print(f"  Reward: {obs.reward}")
    print(f"  Message: {obs.message}")
    print(f"  Precedents returned: {len(obs.precedents)}")
    for p in obs.precedents:
        print(f"    [{p['case_id']}] {p['decision']} — {p['reason'][:50]}")

    # Step 2 — make final decision (use ground truth for testing)
    gt = env._current_episode["ground_truth"]
    expected = gt["expected_action"]
    print(f"\n  Expected action: {expected}")

    action2 = ContentModerationAction(
        action_type=expected,
        reason="Testing with correct action"
    )
    obs = env.step(action2)
    print(f"\nStep 2 — {expected}")
    print(f"  Reward (final score): {obs.reward}")
    print(f"  Done: {obs.done}")
    print(f"  Message: {obs.message}")

    return obs.reward


def main():
    env = ContentModerationEnvironment()

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        score = test_episode(env, task_id)
        scores[task_id] = score

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for tid, s in scores.items():
        print(f"  {tid}: {s}")
    print(f"  Average: {round(sum(scores.values()) / len(scores), 4)}")


if __name__ == "__main__":
    main()