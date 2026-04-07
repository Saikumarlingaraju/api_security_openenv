# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the API Security OpenEnv environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ApiSecurityOpenenvAction(Action):
    """Structured security review submitted by the agent."""

    vulnerabilities: list[str] = Field(
        default_factory=list,
        description="Detected vulnerabilities in the provided API code snippet.",
    )
    severity: str = Field(
        default="",
        description="Overall severity judgement for the task (e.g., CRITICAL, HIGH).",
    )
    fixes: list[str] = Field(
        default_factory=list,
        description="Remediation suggestions for the identified vulnerabilities.",
    )
    rationale: str = Field(
        default="",
        description="Short explanation for the submitted analysis.",
    )


class ApiSecurityOpenenvObservation(Observation):
    """Observation containing task details and grading feedback."""

    task_id: str = Field(default="", description="Unique task identifier.")
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, or hard.")
    objective: str = Field(default="", description="What the agent should accomplish.")
    code_snippet: str = Field(default="", description="Vulnerable API code to audit.")
    attempts_remaining: int = Field(default=0, description="Steps remaining before episode ends.")
    feedback: str = Field(default="", description="Deterministic grader feedback for the last action.")
    last_score: float = Field(default=0.0, description="Latest score from 0.0 to 1.0.")
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Reward components awarded on the last step.",
    )
