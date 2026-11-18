import pathlib
import re

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "arena_evaluation"
ALLOWED = {SOURCE / "ingestion" / "migrate.py"}
FILE_LITERALS = re.compile(r"""["'](power\.yaml|acoustics\.yaml|model_params\.yaml|mobile\.yaml|control\.yaml|component\.yaml|assembly\.yaml)["']""")
SHARE_LOOKUP = re.compile(r"""get_package_share_directory\(\s*["']arena_robots["']\s*\)""")
ROBOTS_DIR = re.compile(r"""(?:/|,)\s*["']robots["']""")


def robot_file_reads(path: pathlib.Path) -> list[str]:
    """Findings of a source file that reads robot definition files by path."""
    text = path.read_text()
    findings = [f"{path}: {match.group(0)}" for match in FILE_LITERALS.finditer(text)]
    if SHARE_LOOKUP.search(text):
        findings += [f"{path}: arena_robots share dir joined with {match.group(0)}" for match in ROBOTS_DIR.finditer(text)]
    return findings


def test_only_the_migration_touches_robot_files():
    findings = [finding for path in sorted(SOURCE.rglob("*.py")) if path not in ALLOWED for finding in robot_file_reads(path)]
    assert findings == []


def test_the_scanner_catches_a_planted_robot_file_read(tmp_path: pathlib.Path):
    planted = tmp_path / "planted.py"
    planted.write_text("import os\nfrom ament_index_python.packages import get_package_share_directory\nshare = get_package_share_directory(\"arena_robots\")\npower = os.path.join(share, \"robots\", \"jackal\", \"telemetry\", \"power.yaml\")\nparams = open(os.path.join(share, \"model_params.yaml\"))\n")
    findings = robot_file_reads(planted)
    assert any("power.yaml" in finding for finding in findings)
    assert any("model_params.yaml" in finding for finding in findings)
    assert any("joined with" in finding for finding in findings)


def test_the_scanner_ignores_a_source_without_robot_file_reads(tmp_path: pathlib.Path):
    clean = tmp_path / "clean.py"
    clean.write_text('share = get_package_share_directory("arena_robots")\nnav2 = share / "config" / "nav2"\nrows = {"robots": []}\n')
    assert robot_file_reads(clean) == []
