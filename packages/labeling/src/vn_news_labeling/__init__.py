"""LLM-assisted summarization labeling pipeline.

Uses Vertex AI (Gemini) to produce summaries; QC layer enforces entity,
number, and NLI-factuality checks. Implementation lands in TICKET-003
(Phase 2).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
