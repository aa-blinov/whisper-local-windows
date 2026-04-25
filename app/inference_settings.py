"""Per-model transcription override settings.

A small, immutable dataclass that captures the inference-time knobs
the user can tweak from the active model card. Lives outside
``model_mapping`` (which is the static registry) and outside the
backends (which only consume the values) so all three layers
import a shared, stable shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class InferenceSettings:
    """Tunables applied to ``WhisperModel.transcribe``.

    Defaults are biased for "new user, recording short voice notes":
    auto-detect language, VAD on (kills the silence-hallucination
    that Whisper is famous for), greedy decoding (``temperature=0``,
    ``beam_size=5`` is faster-whisper's library default).

    GigaAM ignores all of these — its end-to-end transcribe takes no
    such knobs. The settings only get plumbed through to faster-
    whisper-backed models.
    """

    language: Optional[str] = None  # None ⇒ auto-detect
    vad_filter: bool = True
    initial_prompt: Optional[str] = None
    beam_size: int = 5
    temperature: float = 0.0

    # ---- conversion helpers -------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "InferenceSettings":
        """Lenient factory — pull whatever's present in the dict and
        fall back to defaults for missing / wrong-typed entries.
        Used when reading per-alias overrides out of config.yaml."""
        if not data:
            return cls()
        defaults = cls()

        def _opt_str(key: str, fallback: Optional[str]) -> Optional[str]:
            value = data.get(key, fallback)
            if value is None or value == "":
                return None
            return str(value)

        try:
            beam = int(data.get("beam_size", defaults.beam_size))
        except (TypeError, ValueError):
            beam = defaults.beam_size
        beam = max(1, min(beam, 20))

        try:
            temp = float(data.get("temperature", defaults.temperature))
        except (TypeError, ValueError):
            temp = defaults.temperature
        temp = max(0.0, min(temp, 1.0))

        return cls(
            language=_opt_str("language", defaults.language),
            vad_filter=bool(
                data.get("vad_filter", defaults.vad_filter)
            ),
            initial_prompt=_opt_str(
                "initial_prompt", defaults.initial_prompt
            ),
            beam_size=beam,
            temperature=temp,
        )

    def to_mapping(self) -> dict:
        """Plain ``dict`` for persistence — same keys as the schema
        the controller writes into ``model_overrides.<alias>``."""
        return {
            "language": self.language,
            "vad_filter": bool(self.vad_filter),
            "initial_prompt": self.initial_prompt,
            "beam_size": int(self.beam_size),
            "temperature": float(self.temperature),
        }

    def with_change(self, **fields: Any) -> "InferenceSettings":
        """Functional update — keep the dataclass frozen and produce
        a new instance with the supplied fields overridden."""
        return replace(self, **fields)
