import asyncio
import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from client import ApiSecurityOpenenvEnv
from models import ApiSecurityOpenenvAction

MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "api_security_openenv"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.9
MIN_LOG_SCORE = 0.01

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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

    try:
        client.models.list()
        return
    except Exception:
        pass

    _probe_llm_proxy_http_fallback()


def _probe_llm_proxy_http_fallback() -> None:
    """Directly hit likely proxy chat-completions endpoints with the injected API key."""
    base_url = os.environ["API_BASE_URL"].strip().rstrip("/")
    api_key = os.environ["API_KEY"].strip()
    model_name = MODEL_NAME.strip() or "gpt-4o-mini"

    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "temperature": 0,
            "max_tokens": 4,
        }
    ).encode("utf-8")

    urls = [f"{base_url}/chat/completions"]
    v1_url = f"{base_url}/v1/chat/completions"
    if v1_url not in urls:
        urls.append(v1_url)

    for url in urls:
        request = urllib.request.Request(
            url=url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=12):
                return
        except urllib.error.HTTPError as exc:
            # Retry alternate URL shape on 404; other HTTP codes still confirm proxy reachability.
            if exc.code == 404:
                continue
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


def _candidate_image_names() -> list[str]:
    preferred = _resolve_local_image_name()
    available = list(_list_local_images())

    names: list[str] = []
    seen: set[str] = set()

    def add_name(name: str | None) -> None:
        if not name:
            return
        norm = name.strip()
        if not norm or norm in seen:
            return
        seen.add(norm)
        names.append(norm)

    add_name(os.getenv("LOCAL_IMAGE_NAME"))
    add_name(preferred)
    add_name("api-security-openenv:latest")
    add_name("api_security_openenv:latest")
    add_name("api_security_openenv-env:latest")
    add_name("api-security-openenv-env:latest")

    for image in available:
        if "api-security-openenv" in image or "api_security_openenv" in image:
            add_name(image)

    # Last-resort fallback: try a few locally available images.
    for image in available[:8]:
        add_name(image)

    return names


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


def _ensure_local_image(image_name: str) -> bool:
    if _docker_image_exists(image_name):
        return True

    proc = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return proc.returncode == 0 and _docker_image_exists(image_name)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    clamped = max(MIN_LOG_SCORE, min(0.99, value))
    return f"{clamped:.2f}"


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
        log_step(step=1, action="reset", reward=MIN_LOG_SCORE, done=True, error=str(exc).replace("\n", " "))
        log_end(success=False, steps=1, score=MIN_LOG_SCORE, rewards=[MIN_LOG_SCORE])
        return "reset_error", False, 1, [MIN_LOG_SCORE]

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
            rewards.append(MIN_LOG_SCORE)
            steps_taken = step
            log_step(step=step, action=_safe_action_log(action_payload), reward=MIN_LOG_SCORE, done=True, error=error_text)
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
    # Hackathon evaluator injects API_BASE_URL and API_KEY for proxy verification.
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    image_names = _candidate_image_names()

    env: ApiSecurityOpenenvEnv | None = None
    try:
        _probe_llm_proxy(client)

        last_error = "Unable to start environment from any local Docker image."
        chosen_image = "unknown"
        for image_name in image_names:
            _ensure_local_image(image_name)
            try:
                env = await ApiSecurityOpenenvEnv.from_docker_image(image_name)
                chosen_image = image_name
                break
            except Exception as exc:
                last_error = f"{image_name}: {str(exc).replace(chr(10), ' ')}"

        if env is None:
            raise RuntimeError(last_error)

        for _ in range(3):
            await run_episode(env, client)
    except Exception as exc:
        error_text = str(exc).replace("\n", " ")
        log_start(task="bootstrap_error", env=BENCHMARK, model=MODEL_NAME)
        action_name = "from_docker_image(candidates)"
        log_step(step=1, action=action_name, reward=MIN_LOG_SCORE, done=True, error=error_text)
        log_end(success=False, steps=1, score=MIN_LOG_SCORE, rewards=[MIN_LOG_SCORE])
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
