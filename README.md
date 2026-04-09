---
title: API Security OpenEnv Environment
emoji: "🔐"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# API Security Vulnerability Detection Environment

## Motivation

Modern APIs are frequently deployed with authentication gaps, injection vulnerabilities, and sensitive data leaks. This environment simulates practical API security auditing tasks and provides deterministic grading signals suitable for RL-style training and evaluation.

## Environment Summary

The environment presents one vulnerable API snippet per episode. The agent submits a structured security audit action with:

- detected vulnerabilities
- overall severity
- remediation suggestions

The environment returns:

- per-step reward in range [0.0, 1.0]
- deterministic feedback with missing criteria
- episode termination when solved or max steps are reached

## Action Space

`ApiSecurityOpenenvAction`

- `vulnerabilities: list[str]`
- `severity: str`
- `fixes: list[str]`
- `rationale: str`

## Observation Space

`ApiSecurityOpenenvObservation`

- `task_id: str`
- `difficulty: str`
- `objective: str`
- `code_snippet: str`
- `attempts_remaining: int`
- `feedback: str`
- `last_score: float`
- `score_breakdown: dict[str, float]`
- inherited OpenEnv fields: `done`, `reward`, `metadata`

## Tasks and Difficulty

The environment rotates tasks deterministically in this order across resets:

1. `easy_sql_injection`
- Detect SQL injection
- Severity must be `CRITICAL`
- Fix should mention parameterized queries

2. `medium_auth_pii_leak`
- Detect missing authentication and PII exposure
- Severity must be `CRITICAL`
- Fix must include authentication or authorization control

3. `hard_cors_token_leak`
- Detect insecure wildcard CORS with credentials and token leakage
- Severity must be `CRITICAL`
- Fix must mention `httpOnly` cookie handling

## Reward Design

Rewards are deterministic and clamped to [0.0, 1.0].

Per-task weighted rubric:

- Easy: vulnerabilities 0.4, severity 0.3, fix 0.3
- Medium: vulnerabilities 0.5, severity 0.3, fix 0.2
- Hard: vulnerabilities 0.4, severity 0.3, fix 0.3

Additional shaping:

- small penalty for empty/invalid submissions
- small penalty for repeating the same answer
- small progress bonus when a better answer than previous steps is submitted

## Local Setup

### Requirements

- Python 3.10+
- Docker
- `openenv-core`

### Install

```bash
pip install -e .
```

### Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Validate OpenEnv spec

```bash
openenv validate
```

If `openenv` is not on PATH in Windows, use:

```bash
C:/Users/<you>/AppData/Local/Programs/Python/Python313/Scripts/openenv.exe validate
```

## Docker

Build:

```bash
docker build -t api-security-openenv:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 api-security-openenv:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

## Baseline Inference

A mandatory inference script is included at project root:

- `inference.py`

It uses the OpenAI client and required environment variables:

- `API_BASE_URL` (injected by evaluator)
- `MODEL_NAME` (default provided)
- `API_KEY` (injected by evaluator)

Run baseline:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
set API_KEY=your_token_here
python inference.py
```

The script emits strict structured logs:

- `[START]`
- `[STEP]`
- `[END]`

## Reproducibility Notes

- task order is deterministic (`easy -> medium -> hard`)
- grader is deterministic (keyword criteria, no randomness)
- max steps per task is fixed to 3

## Baseline Scores

Fill this section with measured values after running `inference.py` in your setup.

| Task | Baseline Best Reward |
| --- | --- |
| easy_sql_injection | TBD |
| medium_auth_pii_leak | TBD |
| hard_cors_token_leak | TBD |
