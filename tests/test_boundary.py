"""The collector must stay independent of the ML side."""

import ast
from pathlib import Path

COLLECTOR = Path(__file__).resolve().parents[1] / "src" / "flowcast" / "collector"
ALLOWED = ("flowcast.config", "flowcast.collector")


def _imports(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("flowcast"):
            out.append(node.module)
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.startswith("flowcast")]
    return out


def test_collector_imports_nothing_from_ml_side():
    bad = {f"{p.name}: {m}" for p in COLLECTOR.glob("*.py") for m in _imports(p) if not m.startswith(ALLOWED)}
    assert not bad, f"collector imports ML-side modules: {sorted(bad)}"
