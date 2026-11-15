"""Small LiteLLM boundary; importing this module does not load the provider."""

from typing import Protocol


class InvalidLLMResponse(ValueError):
    """The provider returned empty, malformed, or schema-invalid content."""


class LLMClient(Protocol):
    """Text completion dependency shared by the adapter and offline test fakes."""

    def complete(self, *, model: str, prompt: str) -> str: ...


class LiteLLMClient:
    def __init__(self, *, num_retries: int | None = None) -> None:
        self.num_retries = num_retries

    def complete(self, *, model: str, prompt: str) -> str:
        from litellm import completion

        options = {} if self.num_retries is None else {"num_retries": self.num_retries}
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1800,
            **options,
        )
        if not response.choices:
            raise InvalidLLMResponse("LLM returned no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise InvalidLLMResponse("LLM returned an empty response")
        return content
