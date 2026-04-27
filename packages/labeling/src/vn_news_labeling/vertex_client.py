"""Thin wrapper around Vertex AI Gemini for summarization labeling.

We deliberately keep this small and easy to mock in tests:

- :class:`VertexLabeler` holds a single ``GenerativeModel`` instance
  (lazily initialised) and exposes ``generate(system, user) -> str``.
- ``tenacity`` retries transient errors (5xx, 429, network) with
  exponential backoff up to 5 attempts.
- For tests, instantiate with an ``override_callable`` that takes
  ``(system, user)`` and returns the raw JSON string — no Vertex SDK
  required.

The package depends on ``google-cloud-aiplatform``; the ``vertexai``
namespace it exposes is imported lazily so unit tests don't pay the
import cost (and don't need GCP creds).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class VertexLLMError(RuntimeError):
    """Raised when Vertex returns an unrecoverable error."""


class VertexTransientError(RuntimeError):
    """Raised on retryable errors (5xx, 429, transient network)."""


@dataclass(slots=True)
class GenerationParams:
    """Generation parameters mirrored from :class:`prompt.GenerationCfg`."""

    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 256
    response_mime_type: str = "application/json"


_OverrideFn = Callable[[str, str], str]


class VertexLabeler:
    """Vertex AI Gemini wrapper with retry + optional in-process override.

    Parameters
    ----------
    project, location:
        GCP project id + region (e.g. ``asia-southeast1``).
    model_name:
        Gemini model id, e.g. ``gemini-2.0-flash-001``.
    params:
        Generation parameters.
    override_callable:
        If set, the wrapper bypasses Vertex entirely and routes calls
        through this callable. Used in unit + integration tests so we
        never hit the network.
    """

    def __init__(
        self,
        *,
        project: str | None,
        location: str,
        model_name: str,
        params: GenerationParams,
        override_callable: _OverrideFn | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.model_name = model_name
        self.params = params
        self._override = override_callable
        self._model: Any = None  # vertexai.generative_models.GenerativeModel

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._override is not None:
            return None
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT is not set; cannot init Vertex AI"
            raise VertexLLMError(msg)
        # Lazy import — avoids loading the (heavy) google-cloud-aiplatform
        # SDK in unit tests that use the override callable.
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project, location=self.location)
        self._model = GenerativeModel(self.model_name)
        logger.info(
            "Vertex AI initialised: project={} location={} model={}",
            self.project,
            self.location,
            self.model_name,
        )
        return self._model

    @retry(
        reraise=True,
        retry=retry_if_exception_type(VertexTransientError),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
    )
    def generate(self, *, system: str, user: str) -> str:
        """Run one generation. Returns raw text (JSON string in our case).

        Retries up to 5 times on transient failures with exponential
        backoff. Non-transient failures raise :class:`VertexLLMError`.
        """
        if self._override is not None:
            return self._override(system, user)

        model = self._ensure_model()
        from vertexai.generative_models import GenerationConfig

        gen_config = GenerationConfig(
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            max_output_tokens=self.params.max_output_tokens,
            response_mime_type=self.params.response_mime_type,
        )
        try:
            # Gemini supports system_instruction at model construction or
            # per-call; per-call keeps the model object thread-safe-ish.
            from vertexai.generative_models import Content, Part

            request = [
                Content(role="user", parts=[Part.from_text(f"{system}\n\n{user}")]),
            ]
            response = model.generate_content(
                request,
                generation_config=gen_config,
            )
        except Exception as exc:
            text = str(exc).lower()
            transient = any(
                hint in text
                for hint in (
                    "429",
                    "deadline",
                    "unavailable",
                    "internal",
                    "timeout",
                    "resource has been exhausted",
                )
            )
            if transient:
                logger.warning("vertex transient error: {}", exc)
                raise VertexTransientError(str(exc)) from exc
            raise VertexLLMError(str(exc)) from exc

        # Extract text. Gemini SDK returns a candidate list; .text helper
        # joins all parts. If safety filters block the response, .text
        # raises — we surface that as a non-transient error.
        try:
            return str(response.text)
        except Exception as exc:
            raise VertexLLMError(f"empty/blocked response: {exc}") from exc
