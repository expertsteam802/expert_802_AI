"""Public engine facade for Expert 802's native cognitive architecture."""
from .cognition import CognitiveEngine

class ExpertLanguageEngine(CognitiveEngine):
    """Backward-compatible import name used by the HTTP API and tests."""
    pass
