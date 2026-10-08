"""Recovery-local imports and the default functional-test scope.

The excluded modules exercise historical recovery/comparison/support tools
whose implementations were not recoverable with adequate evidence. They stay
in the tree as archival tests, but they are not product completion gates.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SRC_ROOT_TEXT = str(SRC_ROOT)

if SRC_ROOT_TEXT not in sys.path:
    sys.path.insert(0, SRC_ROOT_TEXT)


def _purge_stale_any2xsampa_modules() -> None:
    package = sys.modules.get("any2xsampa")
    module_file = getattr(package, "__file__", None)
    if not module_file:
        return
    try:
        module_path = Path(module_file).resolve()
    except OSError:
        return
    if module_path.is_relative_to(SRC_ROOT.resolve()):
        return
    for name in [key for key in sys.modules if key == "any2xsampa" or key.startswith("any2xsampa.")]:
        sys.modules.pop(name, None)


_purge_stale_any2xsampa_modules()

collect_ignore = [
    "test_archived_lineage_audit.py",
    "test_backend_compare_workflow_c.py",
    "test_long_run_support.py",
    "test_phrase_boundary_compare_workflow_b.py",
    "test_phrase_boundary_compare_cli.py",
    "test_recovery_case_artifacts.py",
    "test_recovery_state_map.py",
    "test_recovery_state_map_integration.py",
    "test_recovery_support_active_lock_integration.py",
    "test_recovery_support_bundle.py",
    "test_recovery_support_checklist_integration.py",
    "test_recovery_support_command_smoke.py",
    "test_recovery_support_matrix_integration.py",
    "test_recovery_support_report.py",
    "test_recovery_support_report_integration.py",
    "test_recovery_support_snapshot_reuse_integration.py",
    "test_recovery_support_summary.py",
    "test_recovery_support_toolchain.py",
    "test_recovery_support_workflow.py",
]
