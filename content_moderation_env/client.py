# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Content Moderation Environment Client."""

from typing import Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ContentModerationAction, ContentModerationObservation


class ContentModerationEnv(
    EnvClient[ContentModerationAction, ContentModerationObservation, State]
):
    """
    Client for the Content Moderation Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with ContentModerationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(options={"task_id": "easy"})
        ...     print(result.observation.post_content)
        ...
        ...     action = ContentModerationAction(
        ...         action_type="retrieve_precedents",
        ...         query="hate speech immigration"
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.precedents)
    """

    def _step_payload(self, action: ContentModerationAction) -> Dict:
        """
        Convert ContentModerationAction to JSON payload for step message.
        """
        payload = {
            "action_type": action.action_type,
        }
        if action.reason is not None:
            payload["reason"] = action.reason
        if action.query is not None:
            payload["query"] = action.query
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ContentModerationObservation]:
        """
        Parse server response into StepResult[ContentModerationObservation].
        """
        obs_data = payload.get("observation", {})

        observation = ContentModerationObservation(
            post_content=obs_data.get("post_content", ""),
            post_metadata=obs_data.get("post_metadata", {}),
            policy_summary=obs_data.get("policy_summary", ""),
            precedents=obs_data.get("precedents", []),
            similar_post_count=obs_data.get("similar_post_count", 0),
            confidence_guidance=obs_data.get("confidence_guidance", ""),
            actions_taken=obs_data.get("actions_taken", []),
            step_count=obs_data.get("step_count", 0),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )