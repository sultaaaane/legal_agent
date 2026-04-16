"""
LLM instances backed by Ollama (local, open-source models).

Swap this file to switch between providers — nothing else in the
codebase needs to change.

Requires:
  pip install langchain-ollama
  ollama pull <model-name>
  ollama serve   (or the Ollama desktop app running)
"""
import os
import json
import re
from typing import Type, TypeVar

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL",   "qwen2:1.5b")
FAST_MODEL       = os.getenv("FAST_MODEL",      "qwen2:1.5b")
FALLBACK_MODEL   = os.getenv("FALLBACK_MODEL",  FAST_MODEL)

PRIMARY_NUM_CTX  = int(os.getenv("PRIMARY_NUM_CTX", "4096"))
FAST_NUM_CTX     = int(os.getenv("FAST_NUM_CTX", "2048"))

ENABLE_OOM_FALLBACK = os.getenv("ENABLE_OOM_FALLBACK", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

# ---------------------------------------------------------------------------
# Base model instances
# ---------------------------------------------------------------------------

_llm_base = ChatOllama(
    model       = PRIMARY_MODEL,
    base_url    = OLLAMA_BASE_URL,
    temperature = 0,
    num_ctx     = PRIMARY_NUM_CTX,
    num_predict = 2048,
)

_llm_fast_base = ChatOllama(
    model       = FAST_MODEL,
    base_url    = OLLAMA_BASE_URL,
    temperature = 0,
    num_ctx     = FAST_NUM_CTX,
    num_predict = 1024,
)


def _is_memory_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "requires more system memory" in text
        or "out of memory" in text
        or "insufficient memory" in text
    )

# ---------------------------------------------------------------------------
# Structured output wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class RobustStructuredLLM:
    """
    Wraps a ChatOllama instance and provides a reliable with_structured_output()
    that falls back to manual JSON extraction when the model drifts from valid JSON.
    """

    def __init__(self, base_llm: ChatOllama):
        self._llm = base_llm
        self._fallback_llm = None

    def with_structured_output(self, schema: Type[T]) -> "_StructuredChain":
        return _StructuredChain(self, schema)

    def invoke(self, messages) -> BaseMessage:
        try:
            return self._llm.invoke(messages)
        except Exception as first_error:
            if not ENABLE_OOM_FALLBACK or not _is_memory_error(first_error):
                raise

            if not FALLBACK_MODEL or FALLBACK_MODEL == getattr(self._llm, "model", None):
                raise

            if self._fallback_llm is None:
                self._fallback_llm = ChatOllama(
                    model       = FALLBACK_MODEL,
                    base_url    = OLLAMA_BASE_URL,
                    temperature = 0,
                    num_ctx     = FAST_NUM_CTX,
                    num_predict = 1024,
                )

            return self._fallback_llm.invoke(messages)

    def bind_tools(self, tools):
        return self._llm.bind_tools(tools)

    def __getattr__(self, name):
        return getattr(self._llm, name)


class _StructuredChain:
    """Runnable that returns a validated Pydantic object from an Ollama call."""

    JSON_INSTRUCTION = (
        "\n\nYou MUST respond with ONLY a valid JSON object that matches the schema below."
        " Do not include markdown code fences, explanations, or any text outside the JSON object."
        " Do not write ```json or ```. Start your response with { and end with }."
    )

    RETRY_INSTRUCTION = (
        "\n\nYour previous response was not valid JSON. "
        "Respond with ONLY a raw JSON object — no markdown, no explanation, "
        "no code fences. Start immediately with { and end with }."
    )

    def __init__(self, wrapped_llm: RobustStructuredLLM, schema: Type[T]):
        self._llm    = wrapped_llm
        self._schema = schema

    def invoke(self, messages: list) -> T:
        augmented = self._inject_schema_hint(messages)

        for attempt in range(3):
            try:
                response = self._llm.invoke(augmented)
                return self._parse(response.content)
            except (ValueError, json.JSONDecodeError) as e:
                if attempt == 2:
                    raise ValueError(
                        f"Failed to get valid JSON from model after 3 attempts.\n"
                        f"Schema: {self._schema.__name__}\n"
                        f"Last error: {e}"
                    )
                augmented = self._inject_retry_hint(augmented, str(e))

    def _inject_schema_hint(self, messages: list) -> list:
        from langchain_core.messages import SystemMessage

        schema_json = json.dumps(self._schema.model_json_schema(), indent=2)
        hint        = self.JSON_INSTRUCTION + f"\n\nJSON Schema:\n{schema_json}"

        result = list(messages)
        for i, msg in enumerate(result):
            if isinstance(msg, SystemMessage):
                result[i] = SystemMessage(content=msg.content + hint)
                return result

        result.insert(0, SystemMessage(content=hint))
        return result

    def _inject_retry_hint(self, messages: list, error: str) -> list:
        from langchain_core.messages import HumanMessage
        result = list(messages)
        result.append(HumanMessage(
            content=f"[Error: {error}]{self.RETRY_INSTRUCTION}"
        ))
        return result

    def _parse(self, content: str) -> T:
        text = content.strip()

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
        text = text.strip()

        # Find outermost JSON object
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in response: {text[:200]}")

        json_str = text[start : end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = self._repair_json(json_str)
            data     = json.loads(json_str)

        return self._schema.model_validate(data)

    @staticmethod
    def _repair_json(text: str) -> str:
        """Fix the most common JSON mistakes made by local models."""
        # Trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text


# ---------------------------------------------------------------------------
# Public exports — same interface as the OpenAI version
# ---------------------------------------------------------------------------

llm      = RobustStructuredLLM(_llm_base)
llm_fast = RobustStructuredLLM(_llm_fast_base)
