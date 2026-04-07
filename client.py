# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Api Security Openenv environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ApiSecurityOpenenvAction, ApiSecurityOpenenvObservation
except ImportError:
    from models import ApiSecurityOpenenvAction, ApiSecurityOpenenvObservation


class ApiSecurityOpenenvEnv(
    EnvClient[ApiSecurityOpenenvAction, ApiSecurityOpenenvObservation, State]
):
    """Typed client for the API security benchmark environment."""

    def _step_payload(self, action: ApiSecurityOpenenvAction) -> Dict:
        """Convert action model to step payload."""
        return {
            "vulnerabilities": action.vulnerabilities,
            "severity": action.severity,
            "fixes": action.fixes,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ApiSecurityOpenenvObservation]:
        """Parse server response payload into typed observation."""
        obs_data = payload.get("observation", {})
        observation = ApiSecurityOpenenvObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            objective=obs_data.get("objective", ""),
            code_snippet=obs_data.get("code_snippet", ""),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            feedback=obs_data.get("feedback", ""),
            last_score=obs_data.get("last_score", 0.0),
            score_breakdown=obs_data.get("score_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state payload."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
