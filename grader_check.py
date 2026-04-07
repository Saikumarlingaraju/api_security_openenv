"""Quick deterministic grader checks for hackathon pre-validation."""

from server.api_security_openenv_environment import ApiSecurityOpenenvEnvironment
from models import ApiSecurityOpenenvAction


GOOD_ACTIONS = {
    "easy_sql_injection": ApiSecurityOpenenvAction(
        vulnerabilities=["SQL injection"],
        severity="CRITICAL",
        fixes=["Use parameterized queries with placeholders."],
        rationale="Direct string interpolation in SQL is unsafe.",
    ),
    "medium_auth_pii_leak": ApiSecurityOpenenvAction(
        vulnerabilities=["Missing authentication", "PII exposure"],
        severity="CRITICAL",
        fixes=["Add authentication and authorization checks."],
        rationale="Anyone can access SSN without auth.",
    ),
    "hard_cors_token_leak": ApiSecurityOpenenvAction(
        vulnerabilities=["Insecure CORS", "token leakage"],
        severity="CRITICAL",
        fixes=["Move token to httpOnly cookie and restrict CORS origins."],
        rationale="Wildcard CORS with credentials plus token leak is dangerous.",
    ),
}


def main() -> None:
    env = ApiSecurityOpenenvEnvironment()

    measured_scores: dict[str, float] = {}

    for _ in range(3):
        reset_obs = env.reset()
        task_id = reset_obs.task_id
        action = GOOD_ACTIONS[task_id]

        result = env.step(action)
        score = float(result.reward or 0.0)

        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score out of range for {task_id}: {score}")

        measured_scores[task_id] = score

    env2 = ApiSecurityOpenenvEnvironment()
    deterministic_scores: dict[str, float] = {}
    for _ in range(3):
        reset_obs = env2.reset()
        task_id = reset_obs.task_id
        deterministic_scores[task_id] = float(env2.step(GOOD_ACTIONS[task_id]).reward or 0.0)

    if measured_scores != deterministic_scores:
        raise ValueError(
            "Determinism check failed. "
            f"first_run={measured_scores}, second_run={deterministic_scores}"
        )

    print("Grader checks passed.")
    for task_id, score in measured_scores.items():
        print(f"{task_id}: {score:.2f}")


if __name__ == "__main__":
    main()
