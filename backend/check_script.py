"""Check model scripts parse the way the UI will read them. Launches nothing.

    venv/bin/python backend/check_script.py config/models/NAME.sh [more.sh ...]

Exit status 0 when every script is configured with no warnings, 1 otherwise.
vLLM scripts are checked against the flag manifest of the engine in their own
exec line, exactly as at startup.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine_flags import install_vllm_parse_manifests  # noqa: E402
from launch_script import parse_script  # noqa: E402

CONFIG_DIR = Path(__file__).parent.parent / "config"


def check(path: Path) -> bool:
    ps = parse_script(path.read_text())
    configured, reason = ps.is_configured()
    problems = list(ps.warnings)
    if not configured:
        problems.append(f"not configured: {reason}")
    if ps.calibrated_gpus is not None and ps.calibrated_gpus != ps.gpu_count:
        problems.append(
            f"GPU count {ps.gpu_count} != calibrated_gpus {ps.calibrated_gpus} "
            f"(calibration will be discarded)"
        )
    mark = "✓" if not problems else "✗"
    print(f"{mark} {path.name}: engine={ps.engine} gpus={ps.gpu_count} port={ps.port}")
    for p in problems:
        print(f"    - {p}")
    return not problems


def main(argv) -> int:
    if not argv:
        print(__doc__.strip())
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    default_vllm = json.loads((CONFIG_DIR / "app.json").read_text()).get("vllm_binary")
    install_vllm_parse_manifests(CONFIG_DIR, lambda: default_vllm)
    results = [check(Path(a)) for a in argv]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
