# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Content Moderation Environment Implementation.

An environment where an LLM agent reviews social media posts,
optionally retrieves precedent cases, and makes moderation decisions
under platform policy constraints.

Tasks:
    easy   — Clear policy violation, single correct action
    medium — Ambiguous content requiring precedent lookup
    hard   — Appeal case with conflicting precedents
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ContentModerationAction, ContentModerationObservation
except ImportError:
    from models import ContentModerationAction, ContentModerationObservation


# ---------------------------------------------------------------------------
# Platform Policy — always visible to agent
# ---------------------------------------------------------------------------

PLATFORM_POLICY = """
PLATFORM COMMUNITY POLICY:

1. REMOVE content that dehumanizes people based on protected characteristics
   (race, religion, ethnicity, gender, sexual orientation, disability).

2. ALLOW political criticism and strong opinions, even if aggressive in tone,
   as long as they target ideas or policies rather than people's identity.

3. ALLOW satire and parody when context makes the intent clear.

4. ALLOW news quotes and factual reporting even on sensitive topics.

5. ADD WARNING LABEL to content that is graphic, disturbing, or potentially
   misleading but does not violate removal policy.

6. ESCALATE any case where the user threatens to leave the platform,
   mentions legal action, or where you are genuinely uncertain.

7. For APPEALS: compare the appealed content against the original removal
   reason and relevant precedents before deciding to overturn or uphold.

8. CONSISTENCY is required — similar content must receive similar decisions.
""".strip()


# ---------------------------------------------------------------------------
# Precedent Database — fixed, 20 cases
# ---------------------------------------------------------------------------

PRECEDENT_DATABASE = {
    "P-001": {
        "content": "These immigrants are destroying our culture and should be deported immediately.",
        "decision": "removed",
        "reason": "Dehumanizing language targeting ethnic group",
        "policy_clause": "protected_characteristics",
        "tags": ["immigration", "ethnicity", "dehumanizing"],
    },
    "P-002": {
        "content": "The government's immigration policy is a complete disaster and needs to change now.",
        "decision": "allowed",
        "reason": "Political criticism of policy, not targeting people",
        "policy_clause": "political_criticism",
        "tags": ["immigration", "policy", "criticism"],
    },
    "P-003": {
        "content": "People of that religion are all terrorists. They shouldn't be allowed here.",
        "decision": "removed",
        "reason": "Collective dehumanization based on religion",
        "policy_clause": "protected_characteristics",
        "tags": ["religion", "terrorism", "dehumanizing"],
    },
    "P-004": {
        "content": "This politician is corrupt and has ruined the economy. Vote them out!",
        "decision": "allowed",
        "reason": "Political opinion targeting a public figure's actions",
        "policy_clause": "political_criticism",
        "tags": ["politics", "corruption", "opinion"],
    },
    "P-005": {
        "content": "lol imagine being that stupid. Darwin awards candidate right here.",
        "decision": "add_warning_label",
        "reason": "Mocking but not targeting protected characteristic",
        "policy_clause": "warning_label",
        "tags": ["mocking", "insult", "non-protected"],
    },
    "P-006": {
        "content": "As reported by BBC: the protest turned violent with 12 arrests.",
        "decision": "allowed",
        "reason": "Factual news reporting",
        "policy_clause": "news_reporting",
        "tags": ["news", "protest", "factual"],
    },
    "P-007": {
        "content": "Women shouldn't be in leadership roles. They're too emotional for that.",
        "decision": "removed",
        "reason": "Gender-based dehumanization",
        "policy_clause": "protected_characteristics",
        "tags": ["gender", "discrimination", "dehumanizing"],
    },
    "P-008": {
        "content": "The CEO's decision to cut jobs was short-sighted and greedy.",
        "decision": "allowed",
        "reason": "Criticism of a business decision by a public figure",
        "policy_clause": "political_criticism",
        "tags": ["business", "criticism", "opinion"],
    },
    "P-009": {
        "content": "This is literally the worst movie ever made. The director has no talent.",
        "decision": "allowed",
        "reason": "Opinion and criticism, not targeting protected characteristic",
        "policy_clause": "political_criticism",
        "tags": ["opinion", "criticism", "entertainment"],
    },
    "P-010": {
        "content": "Gay people are an abomination and shouldn't have rights.",
        "decision": "removed",
        "reason": "Dehumanization based on sexual orientation",
        "policy_clause": "protected_characteristics",
        "tags": ["sexual_orientation", "dehumanizing", "rights"],
    },
    "P-011": {
        "content": "Satirical post: Breaking news — politician discovers consequences exist.",
        "decision": "allowed",
        "reason": "Clear satire targeting a public figure",
        "policy_clause": "satire",
        "tags": ["satire", "politics", "humor"],
    },
    "P-012": {
        "content": "I can't believe how badly the hospital treated my mother. Absolute negligence.",
        "decision": "allowed",
        "reason": "Personal experience and criticism of institution",
        "policy_clause": "political_criticism",
        "tags": ["healthcare", "criticism", "personal"],
    },
    "P-013": {
        "content": "Disabled people are a burden on society and drain public resources.",
        "decision": "removed",
        "reason": "Dehumanization of disabled people",
        "policy_clause": "protected_characteristics",
        "tags": ["disability", "dehumanizing", "protected"],
    },
    "P-014": {
        "content": "This graphic image shows the aftermath of the earthquake. Viewer discretion advised.",
        "decision": "add_warning_label",
        "reason": "Graphic content with legitimate news value",
        "policy_clause": "warning_label",
        "tags": ["graphic", "news", "disaster"],
    },
    "P-015": {
        "content": "Send them back to where they came from. This is our country.",
        "decision": "removed",
        "reason": "Coded dehumanizing language targeting ethnic/immigrant group",
        "policy_clause": "protected_characteristics",
        "tags": ["immigration", "ethnicity", "coded_hate"],
    },
    "P-016": {
        "content": "The immigration numbers this year are at a record high according to official data.",
        "decision": "allowed",
        "reason": "Factual statistical reporting",
        "policy_clause": "news_reporting",
        "tags": ["immigration", "statistics", "factual"],
    },
    "P-017": {
        "content": "I was quoting a historical speech when I wrote that. It is important context.",
        "decision": "allowed",
        "reason": "Historical quote with clear educational framing",
        "policy_clause": "news_reporting",
        "tags": ["historical", "quote", "context"],
    },
    "P-018": {
        "content": "People from that country are lazy and don't want to work.",
        "decision": "removed",
        "reason": "National origin stereotype dehumanizing a group",
        "policy_clause": "protected_characteristics",
        "tags": ["nationality", "stereotype", "dehumanizing"],
    },
    "P-019": {
        "content": "I think the new tax policy unfairly targets middle-class families.",
        "decision": "allowed",
        "reason": "Policy opinion without targeting protected group",
        "policy_clause": "political_criticism",
        "tags": ["tax", "policy", "opinion"],
    },
    "P-020": {
        "content": "This comedian's joke about race is offensive and should be removed.",
        "decision": "add_warning_label",
        "reason": "Edgy comedy — warning label appropriate, not removal",
        "policy_clause": "warning_label",
        "tags": ["comedy", "race", "satire"],
    },
}


# ---------------------------------------------------------------------------
# Episode Data — 15 fixed episodes (5 per task tier)
# ---------------------------------------------------------------------------

EPISODES = {
    "easy": [
        {
            "post_content": "These people are subhuman vermin and should be exterminated from our society. They ruin everything they touch.",
            "post_metadata": {
                "report_count": 47,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-15T14:23:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-001", "P-003", "P-015"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "The president's economic policy has completely destroyed the middle class. This administration is a total failure and needs to go.",
            "post_metadata": {
                "report_count": 3,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-16T09:11:00Z",
            },
            "ground_truth": {
                "expected_action": "allow_content",
                "relevant_precedent_ids": ["P-002", "P-004", "P-019"],
                "policy_clause": "political_criticism",
            },
        },
        {
            "post_content": "Gay people are sick and twisted. They are an abomination and should never be allowed near children.",
            "post_metadata": {
                "report_count": 89,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-17T16:45:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-010", "P-003"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "According to Reuters: Three arrested after protest outside city hall turned violent. Police deployed tear gas.",
            "post_metadata": {
                "report_count": 1,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-18T11:30:00Z",
            },
            "ground_truth": {
                "expected_action": "allow_content",
                "relevant_precedent_ids": ["P-006", "P-016"],
                "policy_clause": "news_reporting",
            },
        },
        {
            "post_content": "Disabled people drain our healthcare system and contribute nothing. Society would be better off without them.",
            "post_metadata": {
                "report_count": 34,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-19T08:00:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-013", "P-001"],
                "policy_clause": "protected_characteristics",
            },
        },
    ],
    "medium": [
        {
            "post_content": "Send them back where they came from. This is our country and our culture. Enough is enough.",
            "post_metadata": {
                "report_count": 12,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-20T13:00:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-015", "P-001", "P-002"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "This comedian's new special has a bit about racial stereotypes. I can't tell if it's clever satire or just offensive. Sharing the clip here.",
            "post_metadata": {
                "report_count": 8,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-21T19:30:00Z",
            },
            "ground_truth": {
                "expected_action": "add_warning_label",
                "relevant_precedent_ids": ["P-011", "P-020", "P-005"],
                "policy_clause": "warning_label",
            },
        },
        {
            "post_content": "Women are too emotional to be effective leaders. That's just biology, not sexism. Look at the data.",
            "post_metadata": {
                "report_count": 19,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-22T10:15:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-007", "P-001"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "I was quoting a 1960s civil rights speech to make a historical point. The words seem harsh today but they are important historical context.",
            "post_metadata": {
                "report_count": 5,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-23T14:00:00Z",
            },
            "ground_truth": {
                "expected_action": "allow_content",
                "relevant_precedent_ids": ["P-017", "P-006"],
                "policy_clause": "news_reporting",
            },
        },
        {
            "post_content": "Graphic aftermath photo from the earthquake in Turkey. Over 2000 dead. This is what our politicians ignore. Warning: disturbing image.",
            "post_metadata": {
                "report_count": 6,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-24T07:45:00Z",
            },
            "ground_truth": {
                "expected_action": "add_warning_label",
                "relevant_precedent_ids": ["P-014", "P-006"],
                "policy_clause": "warning_label",
            },
        },
    ],
    "hard": [
        {
            "post_content": "APPEAL: My post was removed but I was clearly being sarcastic. 'Oh yes, because THAT group of people are SO perfect and never do anything wrong.' Anyone can see that's sarcasm.",
            "post_metadata": {
                "report_count": 0,
                "is_appeal": True,
                "original_removal_reason": "dehumanizing language targeting protected group",
                "platform": "appeal_queue",
                "timestamp": "2024-03-25T09:00:00Z",
            },
            "ground_truth": {
                "expected_action": "uphold_removal",
                "relevant_precedent_ids": ["P-001", "P-011", "P-015"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "APPEAL: I shared a BBC news article about immigration statistics and it was removed. The article is from a mainstream news outlet. This is censorship of factual reporting.",
            "post_metadata": {
                "report_count": 0,
                "is_appeal": True,
                "original_removal_reason": "content promoting anti-immigration sentiment",
                "platform": "appeal_queue",
                "timestamp": "2024-03-26T11:30:00Z",
            },
            "ground_truth": {
                "expected_action": "overturn_removal",
                "relevant_precedent_ids": ["P-006", "P-016", "P-002"],
                "policy_clause": "news_reporting",
            },
        },
        {
            "post_content": "People from that country are just fundamentally different from us. Their values are incompatible with our society. It's not racist to point out cultural differences.",
            "post_metadata": {
                "report_count": 23,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-27T15:00:00Z",
            },
            "ground_truth": {
                "expected_action": "remove_content",
                "relevant_precedent_ids": ["P-018", "P-001", "P-015"],
                "policy_clause": "protected_characteristics",
            },
        },
        {
            "post_content": "APPEAL: My joke post was removed. It said 'Breaking: local politician still hasn't discovered consequences exist.' That's obviously political satire, not a policy violation.",
            "post_metadata": {
                "report_count": 0,
                "is_appeal": True,
                "original_removal_reason": "harassment of public figure",
                "platform": "appeal_queue",
                "timestamp": "2024-03-28T13:45:00Z",
            },
            "ground_truth": {
                "expected_action": "overturn_removal",
                "relevant_precedent_ids": ["P-011", "P-004", "P-009"],
                "policy_clause": "satire",
            },
        },
        {
            "post_content": "I can't believe this platform allows such hate. If you don't remove this content I will be cancelling my subscription and taking legal action. This is discrimination against my community.",
            "post_metadata": {
                "report_count": 2,
                "is_appeal": False,
                "platform": "social_feed",
                "timestamp": "2024-03-29T17:00:00Z",
            },
            "ground_truth": {
                "expected_action": "escalate",
                "relevant_precedent_ids": ["P-012"],
                "policy_clause": "escalation",
            },
        },
    ],
}

MAX_STEPS = 8
TASK_IDS = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Helper — simple keyword search for precedents
# ---------------------------------------------------------------------------

def _search_precedents(query: str, already_retrieved: list, top_k: int = 3) -> list:
    """
    Search precedent database using keyword matching.
    Deterministic — no randomness, no external calls.
    Returns top_k most relevant precedent dicts.
    """
    query_words = set(query.lower().split())
    scored = []

    for pid, case in PRECEDENT_DATABASE.items():
        # Score based on tag overlap + content word overlap
        tag_words = set()
        for tag in case["tags"]:
            tag_words.add(tag.lower())
            tag_words.update(tag.lower().replace("_", " ").replace("-", " ").split())
        content_words = set(case["content"].lower().split())
        reason_words = set(case["reason"].lower().split())

        all_case_words = tag_words | content_words | reason_words
        overlap = len(query_words & all_case_words)
        scored.append((overlap, pid, case))

    # Sort by overlap descending, filter out already retrieved
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = [(s, pid, case) for s, pid, case in scored if pid not in already_retrieved]

    results = []
    for _, pid, case in scored[:top_k]:
        results.append({
            "case_id": pid,
            "content": case["content"],
            "decision": case["decision"],
            "reason": case["reason"],
            "policy_clause": case["policy_clause"],
        })

    return results


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _grade_episode(
    expected_action: str,
    actual_action: str,
    relevant_precedent_ids: list,
    retrieved_precedent_ids: list,
    actions_taken: list,
    step_count: int,
    running_reward: float,
) -> float:
    """
    Deterministic grader — returns final episode score clamped to 0.0–1.0.
    This is what judges see.
    """
    score = running_reward

    # Final decision correctness
    if actual_action == expected_action:
        score += 0.5
    else:
        score -= 0.3

    # Precedent usage — did agent retrieve at least one relevant precedent?
    retrieved_set = set(retrieved_precedent_ids)
    relevant_set = set(relevant_precedent_ids)
    if retrieved_set & relevant_set:
        score += 0.15

    # Clamp to 0.0–1.0
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ContentModerationEnvironment(Environment):
    """
    Content Moderation Environment for OpenEnv.

    The agent reviews social media posts and makes moderation decisions
    under platform policy constraints, optionally retrieving precedent
    cases to support their reasoning.

    Tasks:
        easy   — Clear violation or clear allow, minimal ambiguity
        medium — Ambiguous content requiring precedent lookup
        hard   — Appeal cases with conflicting precedents

    Rewards:
        +0.15 per step for retrieving relevant precedents
        -0.1  per step for retrieving irrelevant precedents
        -0.2  for calling the same action twice (loop penalty)
        -0.3  for hitting step limit without deciding
        +0.5  at end for correct final decision (via grader)
        -0.3  at end for wrong final decision (via grader)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_episode: dict = {}
        self._actions_taken: list = []
        self._retrieved_precedent_ids: list = []
        self._running_reward: float = 0.0
        self._done: bool = False
        self._task_id: str = "easy"

    def reset(self, options: dict = None) -> ContentModerationObservation:
        """
        Reset environment and load a new episode.

        Args:
            options: Optional dict with keys:
                - task_id: "easy", "medium", or "hard" (default: "easy")
        """
        task_id = (options or {}).get("task_id", "easy")
        if task_id not in TASK_IDS:
            task_id = "easy"

        self._task_id = task_id

        # Select episode by index (allows judges to test all episodes)
        episode_index = (options or {}).get("episode_index", 0)
        episode_index = int(episode_index) % len(EPISODES[task_id])
        episode = EPISODES[task_id][episode_index]

        self._current_episode = episode
        self._actions_taken = []
        self._retrieved_precedent_ids = []
        self._running_reward = 0.0
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return ContentModerationObservation(
            post_content=episode["post_content"],
            post_metadata=episode["post_metadata"],
            policy_summary=PLATFORM_POLICY,
            precedents=[],
            actions_taken=[],
            step_count=0,
            message="Episode started. Review the post and make a moderation decision.",
            done=False,
            reward=0.0,
            metadata={"task_id": task_id, "episode_id": self._state.episode_id},
        )

    def step(self, action: ContentModerationAction) -> ContentModerationObservation:
        """
        Execute one moderation action and return updated observation.
        """
        if self._done:
            return self._terminal_observation("Episode already ended.", 0.0)

        self._state.step_count += 1
        step_reward = 0.0
        message = ""

        # Loop penalty — same action called twice
        if action.action_type in self._actions_taken:
            step_reward -= 0.2
            message = f"Penalty: you already called {action.action_type} this episode."

        self._actions_taken.append(action.action_type)

        # Step limit check
        if self._state.step_count >= MAX_STEPS:
            self._done = True
            step_reward -= 0.3
            self._running_reward += step_reward
            final_score = max(0.0, min(1.0, self._running_reward))
            return self._terminal_observation(
                "Step limit reached without a final decision.",
                final_score,
            )

        # --- Handle each action type ---

        if action.action_type == "retrieve_precedents":
            query = action.query or action.reason or "policy violation"
            results = _search_precedents(
                query,
                already_retrieved=self._retrieved_precedent_ids,
            )

            retrieved_ids = [r["case_id"] for r in results]
            self._retrieved_precedent_ids.extend(retrieved_ids)

            # Check relevance
            relevant_ids = self._current_episode["ground_truth"]["relevant_precedent_ids"]
            if any(pid in relevant_ids for pid in retrieved_ids):
                step_reward += 0.15
                message = f"Retrieved {len(results)} precedents. Some are relevant to this case."
            else:
                step_reward -= 0.1
                message = f"Retrieved {len(results)} precedents. None closely match this case."

            self._running_reward += step_reward

            return ContentModerationObservation(
                post_content=self._current_episode["post_content"],
                post_metadata=self._current_episode["post_metadata"],
                policy_summary=PLATFORM_POLICY,
                precedents=results,
                actions_taken=list(self._actions_taken),
                step_count=self._state.step_count,
                message=message,
                done=False,
                reward=step_reward,
                metadata={
                    "task_id": self._task_id,
                    "running_reward": self._running_reward,
                },
            )

        # --- Final decision actions ---
        final_actions = {
            "remove_content",
            "allow_content",
            "add_warning_label",
            "escalate",
            "overturn_removal",
            "uphold_removal",
        }

        if action.action_type in final_actions:
            self._done = True

            expected = self._current_episode["ground_truth"]["expected_action"]
            final_score = _grade_episode(
                expected_action=expected,
                actual_action=action.action_type,
                relevant_precedent_ids=self._current_episode["ground_truth"]["relevant_precedent_ids"],
                retrieved_precedent_ids=self._retrieved_precedent_ids,
                actions_taken=self._actions_taken,
                step_count=self._state.step_count,
                running_reward=self._running_reward,
            )

            if action.action_type == expected:
                message = f"Correct decision: {action.action_type}. Final score: {final_score}"
            else:
                message = f"Incorrect decision: {action.action_type}. Expected: {expected}. Final score: {final_score}"

            return self._terminal_observation(message, final_score)

        # Unknown action type — shouldn't happen given Literal typing
        return ContentModerationObservation(
            post_content=self._current_episode["post_content"],
            post_metadata=self._current_episode["post_metadata"],
            policy_summary=PLATFORM_POLICY,
            precedents=[],
            actions_taken=list(self._actions_taken),
            step_count=self._state.step_count,
            message=f"Unknown action type: {action.action_type}",
            done=False,
            reward=0.0,
            metadata={"task_id": self._task_id},
        )

    def _terminal_observation(self, message: str, final_score: float) -> ContentModerationObservation:
        """Return a done observation with the final score."""
        return ContentModerationObservation(
            post_content=self._current_episode.get("post_content", ""),
            post_metadata=self._current_episode.get("post_metadata", {}),
            policy_summary=PLATFORM_POLICY,
            precedents=[],
            actions_taken=list(self._actions_taken),
            step_count=self._state.step_count,
            message=message,
            done=True,
            reward=final_score,
            metadata={
                "task_id": self._task_id,
                "final_score": final_score,
                "retrieved_precedents": self._retrieved_precedent_ids,
            },
        )

    @property
    def state(self) -> State:
        return self._state