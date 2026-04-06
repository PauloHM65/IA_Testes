"""Converte notacao LaTeX residual para simbolos Unicode."""

import re

from src.steps import register
from src.steps.base import PipelineData
from src.steps.post_processors.base import BasePostProcessor


LATEX_TO_UNICODE = {
    r"\sum": "\u03a3", r"\prod": "\u03a0", r"\int": "\u222b",
    r"\infty": "\u221e", r"\pi": "\u03c0", r"\alpha": "\u03b1", r"\beta": "\u03b2",
    r"\gamma": "\u03b3", r"\delta": "\u03b4", r"\epsilon": "\u03b5", r"\theta": "\u03b8",
    r"\lambda": "\u03bb", r"\mu": "\u03bc", r"\sigma": "\u03c3", r"\omega": "\u03c9",
    r"\phi": "\u03c6", r"\psi": "\u03c8", r"\tau": "\u03c4",
    r"\leq": "\u2264", r"\geq": "\u2265", r"\neq": "\u2260", r"\approx": "\u2248",
    r"\times": "\u00d7", r"\div": "\u00f7", r"\cdot": "\u00b7", r"\pm": "\u00b1",
    r"\sqrt": "\u221a", r"\in": "\u2208", r"\notin": "\u2209",
    r"\subset": "\u2282", r"\supset": "\u2283", r"\cup": "\u222a", r"\cap": "\u2229",
    r"\forall": "\u2200", r"\exists": "\u2203", r"\emptyset": "\u2205",
    r"\rightarrow": "\u2192", r"\leftarrow": "\u2190", r"\Rightarrow": "\u21d2",
    r"\Leftarrow": "\u21d0", r"\leftrightarrow": "\u2194",
    r"\partial": "\u2202", r"\nabla": "\u2207",
}


def latex_to_unicode(text: str) -> str:
    """Converte notacao LaTeX residual para simbolos Unicode."""
    if "\\" not in text and "$" not in text:
        return text

    text = re.sub(r"\\\[|\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\$\$?", "", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\sum_\{([^}]*)\}\^\{([^}]*)\}", r"\u03a3(\1 at\u00e9 \2)", text)
    text = re.sub(r"\\sum_\{([^}]*)\}", r"\u03a3(\1)", text)
    text = re.sub(r"\\prod_\{([^}]*)\}\^\{([^}]*)\}", r"\u03a0(\1 at\u00e9 \2)", text)

    def _superscript(m):
        exp = m.group(1)
        sup_map = {
            "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
            "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
            "n": "\u207f", "i": "\u2071",
        }
        return sup_map.get(exp, f"^{exp}")

    text = re.sub(r"\^\{([^}]*)\}", _superscript, text)
    text = re.sub(r"_\{([^}]*)\}", r"_\1", text)

    for latex_cmd, unicode_char in LATEX_TO_UNICODE.items():
        text = text.replace(latex_cmd, unicode_char)

    text = text.replace("{", "").replace("}", "")
    return text


@register
class LatexToUnicodeProcessor(BasePostProcessor):
    name = "latex_to_unicode"
    label = "Formatando resposta"

    def execute(self, data: PipelineData) -> PipelineData:
        data.resposta = latex_to_unicode(data.resposta)
        return data
