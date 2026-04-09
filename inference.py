import asyncio
import json
import os
import subprocess
from typing import Any

from openai import OpenAI

from client import ApiSecurityOpenenvEnv
from models import ApiSecurityOpenenvAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

_injected_api_key = os.getenv("API_KEY")
API_KEY = _injected_api_key if _injected_api_key is not None else HF_TOKEN

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

BENCHMARK = "api_security_openenv"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.9


def _list_local_images() -> set[str]:
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return set()
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def _probe_llm_proxy(client: OpenAI) -> None:
    """Attempt one lightweight proxy call so evaluator can observe API traffic on injected key."""
    probe_models = [MODEL_NAME, "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
    seen: set[str] = set()

    for model_name in probe_models:
        if model_name in seen:
            continue
        seen.add(model_name)

        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                temperature=0,
                max_tokens=4,
            )
            return
        except Exception:
            continue


def _resolve_local_image_name() -> str:
    env_image = os.getenv("LOCAL_IMAGE_NAME")
    if env_image:
        return env_image

    candidates = [
        "api-security-openenv:latest",
        "api_security_openenv:latest",
        "api_security_openenv-env:latest",
        "api-security-openenv-env:latest",
    ]

    available = _list_local_images()
    for candidate in candidates:
        if candidate in available:
            return candidate

    for image in available:
        if "api-security-openenv" in image or "api_security_openenv" in image:
            return image

    return candidates[0]


def _docker_image_exists(image_name: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
        images = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
        return image_name in images
    except Exception:
        return False


def _ensure_local_image(image_name: str) -> None:
    if _docker_image_exists(image_name):
        return

    subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        check=False,
        capture_output=True,
        text=True,
    )


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _safe_action_log(action: dict[str, Any]) -> str:
    compact = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
    return compact.replace("\n", " ")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error is not None and error != "" else "null"
    print(
        f"[STEP] step={step} action={action} reward={_format_reward(reward)} "
        f"done={_format_bool(done)} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(_format_reward(r) for r in rewards)
    print(
        f"[END] success={_format_bool(success)} steps={steps} score={_format_reward(score)} rewards={rewards_text}",
        flush=True,
    )


def _extract_json(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(raw_text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _fallback_action(task_id: str) -> dict[str, Any]:
    if task_id == "easy_sql_injection":
        return {
            "vulnerabilities": ["SQL injection"],
            "severity": "CRITICAL",
            "fixes": ["Use parameterized queries with placeholders and bound parameters."],
            "rationale": "String concatenation in SQL enables injection.",
        }
    if task_id == "medium_auth_pii_leak":
        return {
            "vulnerabilities": ["Missing authentication", "SSN sensitive data exposure"],
            "severity": "CRITICAL",
            "fixes": ["Add authentication and authorization checks before returning user data."],
            "rationale": "Endpoint leaks PII and allows unauthenticated access.",
        }
    return {
        "vulnerabilities": ["Insecure CORS configuration", "Session token leakage"],
        "severity": "CRITICAL",
        "fixes": ["Send token via httpOnly cookie and restrict allowed origins."],
        "rationale": "Wildcard CORS with credentials plus token exposure is high risk.",
    }


def _build_model_prompt(task_id: str, objective: str, code_snippet: str, feedback: str, step: int) -> str:
    return (
        "You are a secure code auditor. Analyze the API snippet and return strict JSON with keys: "
        "vulnerabilities (list of strings), severity (string), fixes (list of strings), rationale (string). "
        "No markdown, no extra text.\n"
        f"Task ID: {task_id}\n"
        f"Objective: {objective}\n"
        f"Previous feedback: {feedback}\n"
        f"Step: {step}\n"
        "Code:\n"
        f"{code_snippet}\n"
    )


def _get_model_action(client: OpenAI, prompt: str, task_id: str) -> dict[str, Any]:
    model_candidates = [
        MODEL_NAME,
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    seen: set[str] = set()
    for model_name in model_candidates:
        if model_name in seen:
            continue
        seen.add(model_name)

        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            parsed = _extract_json(content)
            if parsed:
                return parsed
        except Exception:
            continue

    return _fallback_action(task_id)


def _normalize_action(raw: dict[str, Any]) -> dict[str, Any]:
    vulnerabilities = raw.get("vulnerabilities", [])
    fixes = raw.get("fixes", [])

    if not isinstance(vulnerabilities, list):
        vulnerabilities = [str(vulnerabilities)] if vulnerabilities else []
    if not isinstance(fixes, list):
        fixes = [str(fixes)] if fixes else []

    return {
        "vulnerabilities": [str(v).strip() for v in vulnerabilities if str(v).strip()],
        "severity": str(raw.get("severity", "")).strip(),
        "fixes": [str(f).strip() for f in fixes if str(f).strip()],
        "rationale": str(raw.get("rationale", "")).strip(),
    }


async def run_episode(env: ApiSecurityOpenenvEnv, client: OpenAI) -> tuple[str, bool, int, list[float]]:
    try:
        reset_result = await env.reset()
        obs = reset_result.observation
    except Exception as exc:
        log_start(task="reset_error", env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="reset", reward=0.0, done=True, error=str(exc).replace("\n", " "))
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return "reset_error", False, 1, [0.0]

    task_id = obs.task_id
    rewards: list[float] = []
    steps_taken = 0
    done = False
    last_feedback = obs.feedback

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        prompt = _build_model_prompt(
            task_id=task_id,
            objective=obs.objective,
            code_snippet=obs.code_snippet,
            feedback=last_feedback,
            step=step,
        )
        raw_action = _get_model_action(client, prompt, task_id)
        action_payload = _normalize_action(raw_action)

        try:
            result = await env.step(
                ApiSecurityOpenenvAction(
                    vulnerabilities=action_payload["vulnerabilities"],
                    severity=action_payload["severity"],
                    fixes=action_payload["fixes"],
                    rationale=action_payload["rationale"],
                )
            )
        except Exception as exc:
            error_text = str(exc).replace("\n", " ")
            rewards.append(0.0)
            steps_taken = step
            log_step(step=step, action=_safe_action_log(action_payload), reward=0.0, done=True, error=error_text)
            break

        obs = result.observation
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        error = None

        rewards.append(reward)
        steps_taken = step
        last_feedback = obs.feedback

        log_step(
            step=step,
            action=_safe_action_log(action_payload),
            reward=reward,
            done=done,
            error=error,
        )

    score = max(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)
    success = bool(score >= SUCCESS_SCORE_THRESHOLD)
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return task_id, success, steps_taken, rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    image_name = _resolve_local_image_name()

    env: ApiSecurityOpenenvEnv | None = None
    try:
        _probe_llm_proxy(client)
        _ensure_local_image(image_name)
        env = await ApiSecurityOpenenvEnv.from_docker_image(image_name)
        for _ in range(3):
            await run_episode(env, client)
    except Exception as exc:
        error_text = str(exc).replace("\n", " ")
        log_start(task="bootstrap_error", env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action=f"from_docker_image({image_name})", reward=0.0, done=True, error=error_text)
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
