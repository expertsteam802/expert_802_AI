"""Client for a self-hosted, OpenAI-compatible open-weight language model.

Set EXPERT802_MODEL_URL to the private inference server's base URL.
The model runs outside Vercel; this module only sends requests to that
user-controlled server and enriches prompts with local dictionary entries.
"""
import json
import os
import re
from urllib.request import Request, urlopen
from .language import dictionary_lookup, tokens

def is_configured():
    return bool(os.environ.get("EXPERT802_MODEL_URL", "").strip())

def configured_model_name():
    return os.environ.get("EXPERT802_MODEL_NAME", "Qwen/Qwen3-1.7B").strip()

def _chat_url():
    base = os.environ.get("EXPERT802_MODEL_URL", "").strip().rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"

def _knowledge_context(message):
    excerpts = []
    seen = set()
    for word in tokens(message):
        if word in seen:
            continue
        seen.add(word)
        entry = dictionary_lookup(word)
        if isinstance(entry, dict):
            definitions = entry.get("definitions", [])
            if definitions:
                excerpts.append(f"{word}: {definitions[0]}")
        elif isinstance(entry, str) and entry:
            excerpts.append(f"{word}: {entry}")
        if len(excerpts) >= 8:
            break
    return "\n".join(excerpts)

def generate_reply(message, history=None):
    system = (
        "You are Expert 802 AI, an independent English-language assistant. "
        "Understand the user's intended meaning from the whole conversation, not just keywords. "
        "Reason carefully, answer naturally in your own words, and use retrieved dictionary facts "
        "as evidence rather than treating them as scripts. Explain uncertainty when evidence is weak. "
        "Do not invent definitions or claim to perform actions on the user's device. "
        "Keep ordinary answers concise, but explain when the user asks for detail."
    )
    knowledge = _knowledge_context(message)
    if knowledge:
        system += "\n\nRelevant entries retrieved from the local English dictionary:\n" + knowledge

    messages = [{"role": "system", "content": system}]
    for item in (history if isinstance(history, list) else [])[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("text", item.get("content", ""))
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:4000]})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != message:
        messages.append({"role": "user", "content": message})

    payload = {
        "model": configured_model_name(),
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 700,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("EXPERT802_MODEL_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(
        _chat_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = float(os.environ.get("EXPERT802_MODEL_TIMEOUT", "45"))
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(answer, list):
        answer = "".join(
            part.get("text", "") for part in answer if isinstance(part, dict)
        )
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The model server returned an empty response.")
    return answer.strip()
