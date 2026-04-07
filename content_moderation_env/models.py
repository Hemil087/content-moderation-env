# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Content Moderation Environment.

Defines the Action and Observation types for a content moderation
agent that reviews posts, retrieves precedents, and makes moderation decisions.
"""

from typing import List, Literal, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ContentModerationAction(Action):
    """
    Action the agent can take in the content moderation environment.

    The agent must choose one of the following action types:
    - retrieve_precedents: Search the precedent database for similar past cases
    - remove_content: Remove the post for policy violation
    - allow_content: Approve the post as policy-compliant
    - add_warning_label: Keep post but attach a context/warning label
    - escalate: Send to senior human moderator
    - overturn_removal: For appeal tasks — reverse a prior removal decision
    - uphold_removal: For appeal tasks — confirm the prior removal was correct
    """

    action_type: Literal[
        "retrieve_precedents",
        "remove_content",
        "allow_content",
        "add_warning_label",
        "escalate",
        "overturn_removal",
        "uphold_removal",
    ] = Field(..., description="The type of moderation action to take")

    reason: Optional[str] = Field(
        default=None,
        description="Reason or justification for the action. Required for all final decisions."
    )

    query: Optional[str] = Field(
        default=None,
        description="Search query for retrieve_precedents action only."
    )


class ContentModerationObservation(Observation):
    """
    Observation returned to the agent after each step.

    Contains everything the agent needs to make a moderation decision:
    the post, platform policy, retrieved precedents, and episode state.
    """

    # Core content — always visible
    post_content: str = Field(
        default="",
        description="The post or comment being reviewed"
    )

    post_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about the post: platform, report_count, timestamp, is_appeal"
    )

    policy_summary: str = Field(
        default="",
        description="Platform policy rules — always visible to agent"
    )

    # Populated only after retrieve_precedents is called
    precedents: List[dict] = Field(
        default_factory=list,
        description="List of similar past cases retrieved from the precedent database"
    )

    # Episode tracking
    actions_taken: List[str] = Field(
        default_factory=list,
        description="History of action_types taken this episode — helps agent avoid loops"
    )

    step_count: int = Field(
        default=0,
        description="Current step number in this episode"
    )

    message: str = Field(
        default="",
        description="Feedback message from environment after last action"
    )

    # Required by OpenEnv spec — must be in observation
    done: bool = Field(
        default=False,
        description="Whether the episode has ended"
    )

    reward: float = Field(
        default=0.0,
        description="Reward received for the last action"
    )

    metadata: dict = Field(
        default_factory=dict,
        description="Additional episode metadata for debugging"
    )