"""Schema detection and conversation normalization shared by data CLIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .common import clean_text


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "question": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "developer": "system",
}

TEXT_KEYS = ("text", "content", "body", "document", "article")


def _string(value: Any) -> str:
    return clean_text(value) if isinstance(value, (str, int, float)) else ""


def normalize_role(value: Any) -> str | None:
    """Map common chat role spellings to ``system``, ``user``, or ``assistant``."""

    if not isinstance(value, str):
        return None
    return ROLE_ALIASES.get(value.strip().lower())


def normalize_messages(value: Any) -> list[dict[str, str]]:
    """Normalize a list of chat messages and merge adjacent identical roles.

    The function accepts OpenAI-style ``role``/``content`` records and
    ShareGPT/OASST-style ``from``/``value`` records.  Empty messages and
    unknown roles are omitted; callers can enforce the required user/assistant
    presence using :func:`validate_messages`.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = normalize_role(item.get("role", item.get("from", item.get("speaker"))))
        content = item.get("content", item.get("value", item.get("text")))
        text = _string(content)
        if role is None or not text:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = f'{normalized[-1]["content"]}\n\n{text}'
        else:
            normalized.append({"role": role, "content": text})
    return normalized


def validate_messages(messages: list[dict[str, str]]) -> tuple[bool, str | None]:
    """Validate the minimum useful requirements for supervised chat data."""

    if not messages:
        return False, "missing_messages"
    if not any(message["role"] == "user" for message in messages):
        return False, "missing_user"
    if not any(message["role"] == "assistant" for message in messages):
        return False, "missing_assistant"
    if messages[0]["role"] == "assistant":
        return False, "assistant_starts_conversation"
    return True, None


def normalize_dolly_record(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert a Dolly-style instruction/context/response record to messages."""

    instruction = _string(record.get("instruction", record.get("prompt")))
    response = _string(record.get("response", record.get("completion", record.get("answer"))))
    context = _string(record.get("context"))
    if not instruction or not response:
        return []
    user_content = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": response}]


def record_to_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert Dolly, OASST, ShareGPT, or normalized chat records to messages."""

    if isinstance(record.get("messages"), Sequence) and not isinstance(record.get("messages"), (str, bytes)):
        return normalize_messages(record["messages"])
    if isinstance(record.get("conversations"), Sequence) and not isinstance(record.get("conversations"), (str, bytes)):
        return normalize_messages(record["conversations"])
    if isinstance(record.get("conversation"), Sequence) and not isinstance(record.get("conversation"), (str, bytes)):
        return normalize_messages(record["conversation"])
    if normalize_role(record.get("role", record.get("from"))) is not None:
        single = normalize_messages([record])
        if single:
            return single
    return normalize_dolly_record(record)


def detect_schema(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Infer a useful high-level schema from a sample of records."""

    keys: set[str] = set()
    message_keys: set[str] = set()
    for record in records:
        keys.update(str(key) for key in record.keys())
        for container_key in ("messages", "conversations", "conversation"):
            value = record.get(container_key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for message in value:
                    if isinstance(message, Mapping):
                        message_keys.update(str(key) for key in message.keys())
    if "messages" in keys or "conversations" in keys or "conversation" in keys or message_keys.intersection({"role", "from", "speaker"}):
        kind = "conversation"
    elif {"instruction", "response"}.issubset(keys):
        kind = "dolly"
    elif "text" in keys or "content" in keys:
        kind = "text"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "record_keys": sorted(keys),
        "message_keys": sorted(message_keys),
        "text_keys": [key for key in TEXT_KEYS if key in keys],
    }


def record_to_text(record: Mapping[str, Any]) -> str:
    """Extract a plain-text training document from a generic record."""

    messages = record_to_messages(record)
    if messages:
        return "\n\n".join(f'{message["role"]}: {message["content"]}' for message in messages)
    for key in TEXT_KEYS:
        text = _string(record.get(key))
        if text:
            return text
    instruction = _string(record.get("instruction", record.get("prompt")))
    response = _string(record.get("response", record.get("completion")))
    return "\n\n".join(part for part in (instruction, response) if part)


def render_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Render normalized messages using stable role markers for SFT."""

    rendered: list[str] = []
    for message in messages:
        rendered.append(f'<|{message["role"]}|>\n{message["content"]}\n')
    return "".join(rendered)
