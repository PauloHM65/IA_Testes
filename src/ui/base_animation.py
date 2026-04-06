"""Base para animacoes de terminal."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod


class BaseAnimation(ABC):
    """Animacao em thread separada. Subclasses implementam _loop()."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._message = ""

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._clear_line()

    def update(self, msg: str):
        self._message = msg

    def _run(self):
        while not self._stop_event.is_set():
            self._loop()
            time.sleep(0.1)

    @abstractmethod
    def _loop(self):
        """Renderiza um frame da animacao."""
        ...

    @staticmethod
    def _clear_line():
        print("\r" + " " * 60 + "\r", end="", flush=True)

    @staticmethod
    def _render(text: str):
        print(f"\r{text}          ", end="", flush=True)
