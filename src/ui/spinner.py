"""Spinner animado com braille characters."""

from src.ui.base_animation import BaseAnimation

FRAMES = ["\u28fe", "\u28fd", "\u28fb", "\u28bf", "\u287f", "\u28df", "\u28ef", "\u28f7"]


class SpinnerAnimation(BaseAnimation):
    """Spinner: ⣾ ⣽ ⣻ ⢿ ⡿ ⣟ ⣯ ⣷"""

    def __init__(self, color: str = "\033[38;5;208m", reset: str = "\033[0m"):
        super().__init__()
        self._frame_idx = 0
        self._color = color
        self._reset = reset

    def _loop(self):
        frame = FRAMES[self._frame_idx % len(FRAMES)]
        msg = self._message or "Pensando..."
        self._render(f"{self._color}{frame} {msg}{self._reset}")
        self._frame_idx += 1
