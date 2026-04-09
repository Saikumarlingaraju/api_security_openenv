# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""API Security OpenEnv environment implementation."""

from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ApiSecurityOpenenvAction, ApiSecurityOpenenvObservation
except ImportError:
    from models import ApiSecurityOpenenvAction, ApiSecurityOpenenvObservation


@dataclass(frozen=True)
class TaskSpec:
    """Deterministic task definition used by the grader."""

    task_id: str
    difficulty: str
    objective: str
    code_snippet: str
    expected_severity: str
    vulnerability_aliases: dict[str, tuple[str, ...]]
    fix_aliases: dict[str, tuple[str, ...]]
    weights: dict[str, float]
    require_all_vulnerabilities_for_credit: bool = True
    require_all_fixes_for_credit: bool = False
    max_steps: int = 3


TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="easy_sql_injection",
        difficulty="easy",
        objective="Identify SQL Injection, set severity to CRITICAL, and propose a safe query fix.",
        code_snippet=(
            "@app.route('/search')\n"
            "def search():\n"
            "    query = request.args.get('q')\n"
            "    results = db.execute(f\"SELECT * FROM users WHERE name = '{query}'\")\n"
            "    return results"
        ),
        expected_severity="critical",
        vulnerability_aliases={
            "sql_injection": (
                "sql injection",
                "sqli",
                "unsafe sql",
                "query injection",
            )
        },
        fix_aliases={
            "parameterized_query": (
                "parameterized",
                "prepared statement",
                "bind parameter",
                "placeholder",
            )
        },
        weights={"vulnerabilities": 0.4, "severity": 0.3, "fixes": 0.3},
        require_all_vulnerabilities_for_credit=True,
        require_all_fixes_for_credit=False,
    ),
    TaskSpec(
        task_id="medium_auth_pii_leak",
        difficulty="medium",
        objective=(
            "Identify missing authentication and PII exposure (SSN), set severity to CRITICAL, "
            "and include authentication in the remediation plan."
        ),
        code_snippet=(
            "@app.route('/api/user/<user_id>')\n"
            "def get_user(user_id):\n"
            "    user = User.query.get(user_id)\n"
            "    return jsonify({\n"
            "        'email': user.email,\n"
            "        'phone': user.phone,\n"
            "        'ssn': user.ssn\n"
            "    })"
        ),
        expected_severity="critical",
        vulnerability_aliases={
            "missing_authentication": (
                "missing authentication",
                "no authentication",
                "auth bypass",
                "unauthenticated access",
                "missing authorization",
            ),
            "pii_exposure": (
                "pii exposure",
                "sensitive data exposure",
                "ssn exposed",
                "information disclosure",
                "data leak",
            ),
        },
        fix_aliases={
            "auth_control": (
                "require_auth",
                "authentication",
                "authorization check",
                "access control",
                "jwt validation",
            )
        },
        weights={"vulnerabilities": 0.5, "severity": 0.3, "fixes": 0.2},
        require_all_vulnerabilities_for_credit=True,
        require_all_fixes_for_credit=False,
    ),
    TaskSpec(
        task_id="hard_cors_token_leak",
        difficulty="hard",
        objective=(
            "Identify insecure CORS credentials wildcard and token leakage, set severity to CRITICAL, "
            "and recommend storing tokens in httpOnly cookies."
        ),
        code_snippet=(
            "@app.route('/api/data')\n"
            "def get_data():\n"
            "    response.headers['Access-Control-Allow-Origin'] = '*'\n"
            "    response.headers['Access-Control-Allow-Credentials'] = 'true'\n"
            "    return jsonify({\n"
            "        'data': data,\n"
            "        'token': generate_session_token()\n"
            "    })"
        ),
        expected_severity="critical",
        vulnerability_aliases={
            "insecure_cors": (
                "insecure cors",
                "cors misconfiguration",
                "wildcard cors",
                "allow-origin *",
                "credentials true with wildcard",
            ),
            "token_leakage": (
                "token leakage",
                "token exposed",
                "sensitive token in response",
                "credential exposure",
                "session token leak",
            ),
        },
        fix_aliases={
            "httponly_cookie": (
                "httponly cookie",
                "http only cookie",
                "secure cookie",
                "set-cookie",
            )
        },
        weights={"vulnerabilities": 0.4, "severity": 0.3, "fixes": 0.3},
        require_all_vulnerabilities_for_credit=True,
        require_all_fixes_for_credit=False,
    ),
]

OPEN_INTERVAL_MIN_SCORE = 0.01
OPEN_INTERVAL_MAX_SCORE = 0.99


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _to_open_interval(score: float) -> float:
    """Map a clamped [0,1] score into strict open interval (0,1)."""
    clamped = max(0.0, min(1.0, score))
    return OPEN_INTERVAL_MIN_SCORE + (OPEN_INTERVAL_MAX_SCORE - OPEN_INTERVAL_MIN_SCORE) * clamped


class ApiSecurityOpenenvEnvironment(Environment):
    """Environment for auditing vulnerable API snippets."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_cursor = 0
        self._current_task = TASKS[0]
        self._steps_in_episode = 0
        self._best_score = 0.0
        self._last_feedback = ""
        self._last_breakdown: dict[str, float] = {}
        self._last_action_signature = ""

    def reset(self) -> ApiSecurityOpenenvObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = TASKS[self._task_cursor % len(TASKS)]
        self._task_cursor += 1
        self._steps_in_episode = 0
        self._best_score = 0.0
        self._last_feedback = "Submit vulnerabilities, severity, and fixes to be graded."
        self._last_breakdown = {}
        self._last_action_signature = ""

        return self._build_observation(
            done=False,
            reward=0.0,
            score=0.0,
            feedback=self._last_feedback,
            breakdown=self._last_breakdown,
        )

    def step(self, action: ApiSecurityOpenenvAction) -> ApiSecurityOpenenvObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._steps_in_episode += 1

        score, breakdown, feedback = self._grade_action(action)

        signature = _normalize(
            "|".join(action.vulnerabilities)
            + "#"
            + action.severity
            + "#"
            + "|".join(action.fixes)
        )
        if signature and signature == self._last_action_signature:
            score = max(0.0, score - 0.05)
            breakdown["repetition_penalty"] = -0.05
            feedback += " Repeated answer penalty applied."
        self._last_action_signature = signature

        if score > self._best_score:
            progress_bonus = min(0.05, score - self._best_score)
            score = min(1.0, score + progress_bonus)
            if progress_bonus > 0.0:
                breakdown["progress_bonus"] = progress_bonus
            self._best_score = score

        score = _to_open_interval(score)

        done = score >= 0.95 or self._steps_in_episode >= self._current_task.max_steps
        self._last_feedback = feedback
        self._last_breakdown = breakdown

        return self._build_observation(
            done=done,
            reward=score,
            score=score,
            feedback=feedback,
            breakdown=breakdown,
        )

    def _grade_action(self, action: ApiSecurityOpenenvAction) -> tuple[float, dict[str, float], str]:
        task = self._current_task
        submitted_vulns = [_normalize(v) for v in action.vulnerabilities]
        submitted_fixes = [_normalize(f) for f in action.fixes]
        submitted_severity = _normalize(action.severity)

        breakdown = {
            "vulnerabilities": 0.0,
            "severity": 0.0,
            "fixes": 0.0,
            "invalid_action_penalty": 0.0,
        }

        vuln_hits: list[str] = []
        missing_vulns: list[str] = []
        for vuln_name, aliases in task.vulnerability_aliases.items():
            found = any(any(alias in pred for alias in aliases) for pred in submitted_vulns)
            if found:
                vuln_hits.append(vuln_name)
            else:
                missing_vulns.append(vuln_name)

        if task.require_all_vulnerabilities_for_credit:
            if not missing_vulns:
                breakdown["vulnerabilities"] = task.weights["vulnerabilities"]
        else:
            total = max(1, len(task.vulnerability_aliases))
            breakdown["vulnerabilities"] = task.weights["vulnerabilities"] * (len(vuln_hits) / total)

        if submitted_severity == task.expected_severity:
            breakdown["severity"] = task.weights["severity"]

        fix_hits = 0
        missing_fixes: list[str] = []
        for fix_name, aliases in task.fix_aliases.items():
            found = any(any(alias in pred for alias in aliases) for pred in submitted_fixes)
            if found:
                fix_hits += 1
            else:
                missing_fixes.append(fix_name)

        if task.require_all_fixes_for_credit:
            if not missing_fixes:
                breakdown["fixes"] = task.weights["fixes"]
        else:
            total = max(1, len(task.fix_aliases))
            breakdown["fixes"] = task.weights["fixes"] * (fix_hits / total)

        if not submitted_vulns and not submitted_fixes and not submitted_severity:
            breakdown["invalid_action_penalty"] = -0.1

        score = sum(breakdown.values())
        score = max(0.0, min(1.0, score))

        missing_vuln_text = ", ".join(missing_vulns) if missing_vulns else "none"
        missing_fix_text = ", ".join(missing_fixes) if missing_fixes else "none"
        feedback = (
            f"Task {task.task_id}: vuln_hits={len(vuln_hits)}/{len(task.vulnerability_aliases)}, "
            f"severity_match={'yes' if breakdown['severity'] > 0 else 'no'}, "
            f"missing_vulns={missing_vuln_text}, missing_fixes={missing_fix_text}."
        )
        return score, breakdown, feedback

    def _build_observation(
        self,
        *,
        done: bool,
        reward: float,
        score: float,
        feedback: str,
        breakdown: dict[str, float],
    ) -> ApiSecurityOpenenvObservation:
        attempts_remaining = max(0, self._current_task.max_steps - self._steps_in_episode)
        return ApiSecurityOpenenvObservation(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            objective=self._current_task.objective,
            code_snippet=self._current_task.code_snippet,
            attempts_remaining=attempts_remaining,
            feedback=feedback,
            last_score=score,
            score_breakdown=breakdown,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "task_sequence_index": self._task_cursor,
            },
        )

    @property
    def state(self) -> State:
        return self._state
