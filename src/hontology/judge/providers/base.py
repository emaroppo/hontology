"""Provider-agnostic completion interface.

Every judge call in the system goes through :class:`ChatProvider`. The point of
the indirection is that the pipeline, the prompt registry and the evaluation
layer all stay ignorant of who served the tokens — a run configured for a local
model and the same run configured for a hosted one differ in exactly one config
field, and nothing downstream changes.

Note which fields are *behavior* and which are *transport*. ``provider`` and
``model`` and everything in :class:`GenerationConfig` change the output, so they
belong in a run's config hash. Hosts, ports, API keys and timeouts do not: the
same request served from a different machine is the same experiment.
:meth:`GenerationConfig.hashable` draws that line explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings. Every field here is behavior and is hashed.

    ``timeout_s`` is the one exception and is excluded from :meth:`hashable`:
    waiting longer for the same tokens does not make a different experiment.
    """

    temperature: float = 0.0
    context_window: int = 8192
    max_output_tokens: int | None = None
    seed: int | None = None
    timeout_s: float = 600.0

    def hashable(self) -> dict:
        d = asdict(self)
        d.pop("timeout_s")
        return d


@dataclass(frozen=True)
class Completion:
    """One provider response, normalized across backends.

    ``reasoning`` holds a thinking trace when the model emitted one into a
    dedicated channel, and is empty otherwise. It is persisted alongside the
    verdict for error analysis but never parsed for the answer.
    """

    text: str
    reasoning: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0
    stop_reason: str | None = None


class ProviderError(RuntimeError):
    """A call failed in a way the caller should record, not crash on.

    Judge runs turn one of these into an error row for the pair and carry on, so
    a single unparseable response never costs a whole run. ``retryable`` marks
    transport faults (timeouts, 5xx) as distinct from a refusal or a bad request.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class ChatProvider(Protocol):
    """What the judge needs from a text-generation backend."""

    name: str

    def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        config: GenerationConfig,
        want_json: bool = True,
        want_reasoning: bool = False,
    ) -> Completion:
        """Run one completion.

        ``want_json`` asks the backend to constrain output to valid JSON where it
        can. ``want_reasoning`` asks for a thinking trace. Backends that cannot
        honor a flag must degrade to prompt-level instruction rather than raise —
        the caller always parses defensively regardless.
        """
        ...

    def health(self) -> str:
        """Return a short human-readable status, or raise :class:`ProviderError`."""
        ...
