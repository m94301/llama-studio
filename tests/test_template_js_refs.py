"""Every function the page templates call must be defined somewhere the page loads.

A function deleted by mistake (autoScrollLogViewer, lost in an edit of the log
viewer code) fails silently in the browser: the call throws inside a caught
fetch loop and the page just stops doing that one thing. This catches it here.
"""

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "frontend" / "templates"

# Browser and language globals the templates use.
BUILTINS = set("""
if for while return switch catch function typeof new await async
setTimeout clearTimeout setInterval clearInterval requestAnimationFrame
parseInt parseFloat isNaN encodeURIComponent decodeURIComponent confirm alert fetch
String Number Boolean Math JSON Object Array Date Promise Set Map RegExp Error
URL URLSearchParams FormData WebSocket AbortController TextDecoder
""".split())


def _texts():
    return {p.relative_to(TEMPLATES).as_posix(): p.read_text() for p in TEMPLATES.rglob("*.html")}


def _defined(all_text: str) -> set:
    return (
        set(re.findall(r"function\s+([A-Za-z_]\w*)\s*\(", all_text))
        | set(re.findall(r"window\.([A-Za-z_]\w*)\s*=", all_text))
        | set(re.findall(r"(?:const|let|var)\s+([A-Za-z_]\w*)\s*=", all_text))
    )


def _calls(code: str) -> set:
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"`[^`]*`|'[^'\n]*'|\"[^\"\n]*\"", "''", code)
    return set(re.findall(r"(?<![.\w$])([A-Za-z_]\w*)\s*\(", code))


def test_every_called_function_is_defined():
    texts = _texts()
    defined = _defined("".join(texts.values())) | BUILTINS
    missing = {}
    for name, text in texts.items():
        handlers = re.findall(r'(?:onclick|onchange|oninput|onsubmit|hx-on[:\w-]*)\s*=\s*"([^"]*)"', text)
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", text, re.S)
        for code in handlers + scripts:
            for fn in _calls(code) - defined:
                missing.setdefault(fn, set()).add(name)
    assert not missing, f"called but never defined: {missing}"
