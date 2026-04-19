from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.outputs import Generation, LLMResult

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, LLM_SEED, LLM_TIMEOUT_SECONDS


class GroqClientError(RuntimeError):
    pass


def _build_headers() -> dict[str, str]:
    if not GROQ_API_KEY:
        raise GroqClientError("GROQ_API_KEY не задан. Добавь ключ в .env или переменные окружения.")

    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


def build_chat_messages(prompt: str, system_message: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_groq_request_payload(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "seed": LLM_SEED,
    }
    if stop:
        payload["stop"] = stop
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": False,
                "schema": response_schema,
            },
        }
    return payload


def _parse_response_content(payload: dict[str, Any]) -> str:
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise GroqClientError("Groq API вернул неожиданный формат ответа.") from exc


def groq_chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_groq_request_payload(
        messages=messages,
        temperature=temperature,
        stop=stop,
        response_schema=response_schema,
    )

    try:
        response = httpx.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=_build_headers(),
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400]
        raise GroqClientError(f"Groq API вернул ошибку {exc.response.status_code}: {body}") from exc
    except httpx.HTTPError as exc:
        raise GroqClientError(f"Ошибка сетевого запроса к Groq API: {exc}") from exc

    return response.json()


async def agroq_chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_groq_request_payload(
        messages=messages,
        temperature=temperature,
        stop=stop,
        response_schema=response_schema,
    )

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=_build_headers(),
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400]
        raise GroqClientError(f"Groq API вернул ошибку {exc.response.status_code}: {body}") from exc
    except httpx.HTTPError as exc:
        raise GroqClientError(f"Ошибка сетевого запроса к Groq API: {exc}") from exc

    return response.json()


def generate_text_from_prompt(
    prompt: str,
    *,
    temperature: float = 0.0,
    system_message: str | None = None,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    payload = groq_chat_completion(
        messages=build_chat_messages(prompt, system_message=system_message),
        temperature=temperature,
        stop=stop,
        response_schema=response_schema,
    )
    content = _parse_response_content(payload)
    if not content:
        raise GroqClientError("Groq API вернул пустой текст ответа.")
    return content


async def agenerate_text_from_prompt(
    prompt: str,
    *,
    temperature: float = 0.0,
    system_message: str | None = None,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    payload = await agroq_chat_completion(
        messages=build_chat_messages(prompt, system_message=system_message),
        temperature=temperature,
        stop=stop,
        response_schema=response_schema,
    )
    content = _parse_response_content(payload)
    if not content:
        raise GroqClientError("Groq API вернул пустой текст ответа.")
    return content


def generate_llm_result(
    prompt: str,
    *,
    n: int = 1,
    temperature: float = 0.01,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> LLMResult:
    generations = [
        Generation(
            text=generate_text_from_prompt(
                prompt,
                temperature=temperature,
                stop=stop,
                response_schema=response_schema,
            )
        )
        for _ in range(n)
    ]
    return LLMResult(generations=[generations])


async def agenerate_llm_result(
    prompt: str,
    *,
    n: int = 1,
    temperature: float = 0.01,
    stop: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> LLMResult:
    generations = [
        Generation(
            text=await agenerate_text_from_prompt(
                prompt,
                temperature=temperature,
                stop=stop,
                response_schema=response_schema,
            )
        )
        for _ in range(n)
    ]
    return LLMResult(generations=[generations])


def parse_json_response(text: str) -> dict[str, Any]:
    return json.loads(text)
