from __future__ import annotations

import json
import os
import sys

from openai import OpenAI


def parse_json_response(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def main() -> int:
    base_url = os.environ.get("LOCAL_VLM_BASE_URL", "http://127.0.0.1:18000/v1")
    model = os.environ.get("LOCAL_VLM_MODEL", "qwen2.5-vl-7b-instruct")
    client = OpenAI(api_key=os.environ.get("LOCAL_VLM_API_KEY", "local"), base_url=base_url, timeout=60)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown.",
            },
            {
                "role": "user",
                "content": (
                    "Choose one navigation skill from GEOMETRIC_EXPLORE and FALLBACK_APEXNAV. "
                    "Return JSON with selected_skill and confidence."
                ),
            },
        ],
        temperature=0,
        max_tokens=128,
        extra_body={"repetition_penalty": 1.05},
    )
    text = response.choices[0].message.content or ""
    print(text)
    try:
        data = parse_json_response(text)
    except json.JSONDecodeError:
        print("invalid_json", file=sys.stderr)
        return 1
    if data.get("selected_skill") not in {"GEOMETRIC_EXPLORE", "FALLBACK_APEXNAV"}:
        print("unexpected_skill", data.get("selected_skill"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
