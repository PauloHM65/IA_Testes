"""Log de eficiencia: decorator @timed_step + PerformanceLogger."""

from __future__ import annotations

import functools
import threading
import time
from datetime import datetime
from pathlib import Path

PERF_DIR = Path(__file__).parent.parent.parent / "logs" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)

PERF_FILE = PERF_DIR / "efficiency.txt"


# ---------------------------------------------------------------------------
# Decorator: cronometra execute() e grava tempos em data.timings
# ---------------------------------------------------------------------------

def timed_step(execute_fn):
    """Decorator para BaseStep.execute — mede tempo e salva em data.timings."""

    @functools.wraps(execute_fn)
    def wrapper(self, data):
        step_name = getattr(self, "label", None) or getattr(self, "name", execute_fn.__qualname__)
        start = time.perf_counter()
        result = execute_fn(self, data)
        elapsed = time.perf_counter() - start
        result.timings[step_name] = elapsed
        return result

    return wrapper


# ---------------------------------------------------------------------------
# PerformanceLogger: escreve relatorio de eficiencia em arquivo .txt
# ---------------------------------------------------------------------------

class PerformanceLogger:
    """Grava metricas de tempo de cada interacao no arquivo de eficiencia."""

    def __init__(self):
        self._lock = threading.Lock()

    def log_interaction(
        self,
        *,
        service: str,
        pergunta: str,
        timings: dict[str, float],
        total_time: float,
        model: str = "",
        provider: str = "",
    ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = []

        lines.append("=" * 70)
        lines.append(f"  INTERACAO  |  {now}")
        lines.append("=" * 70)
        lines.append(f"  Servico : {service}")
        lines.append(f"  Modelo  : {provider}/{model}")
        lines.append(f"  Pergunta: \"{pergunta}\"")
        lines.append("-" * 70)
        lines.append(f"  {'ETAPA':<35} {'TEMPO':>10}  {'%':>6}")
        lines.append("-" * 70)

        for step_name, elapsed in timings.items():
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            lines.append(f"  {step_name:<35} {elapsed:>9.3f}s  {pct:>5.1f}%")

        lines.append("-" * 70)
        lines.append(f"  {'TOTAL':<35} {total_time:>9.3f}s  100.0%")
        lines.append("=" * 70)
        lines.append("")

        text = "\n".join(lines) + "\n"

        with self._lock:
            with open(PERF_FILE, "a", encoding="utf-8") as f:
                f.write(text)


# Singleton
_perf_logger = PerformanceLogger()


def get_perf_logger() -> PerformanceLogger:
    return _perf_logger