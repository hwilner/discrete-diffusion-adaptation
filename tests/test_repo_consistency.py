"""Consistency checks between the repository and its own documentation.

The repository states facts about itself in several places: the version
of the package, and the size of the test suite. Those statements were
allowed to drift away from the truth, so this module pins them.

Two checks live here:

1. The version in ``pyproject.toml`` must equal ``dida.__version__``.
   A release built from the mismatched pair would ship one version while
   the code reports another.
2. The number of tests the documentation claims must equal the number of
   tests the suite actually collects. The count is taken from a real
   collection run in a subprocess, so it cannot drift silently.

Both checks are about the repository rather than about any scientific
result: they make no claim about the behaviour of the DiDA mechanism.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import dida

# Repository root: the parent of the ``tests`` directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where each documented test-count claim lives, and how to read it. Each
# entry maps a file to the regular expression with one capture group that
# holds the documented number.
COUNT_CLAIMS = (
    ("README.md", r"\*\*(\d+) unit and integration tests\*\*"),
    ("CONTRIBUTING.md", r"all (\d+) tests should pass"),
    ("docs/TESTING.md", r"All (\d+) tests should pass"),
    ("results.md", r"\| Tests collected \| \*\*(\d+)\*\* \|"),
)


def _documented_count(relative_path, pattern):
    """Read the test count documented in one file.

    Args:
        relative_path: Path of the file, relative to the repository root.
        pattern: Regular expression with one capture group holding the
            documented number.

    Returns:
        The documented number as an int.

    Raises:
        AssertionError: If the file does not document the count exactly
            once, which means the claim was reworded and the check needs
            to be updated alongside it.
    """
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    matches = re.findall(pattern, text)
    assert len(matches) == 1, (
        "%s must document the test count exactly once (found %d matches of "
        "%r); update this check together with the wording."
        % (relative_path, len(matches), pattern)
    )
    return int(matches[0])


def _collected_test_count():
    """Count the tests pytest collects for the whole suite.

    The count comes from a real collection run in a subprocess, so it
    counts parametrized cases as pytest would run them and does not
    depend on which tests the current process happens to run.

    Returns:
        The number of collected test items.
    """
    env = dict(os.environ)
    search_path = (
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        env.get("PYTHONPATH", ""),
    )
    env["PYTHONPATH"] = os.pathsep.join(path for path in search_path if path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, (
        "collecting the suite failed:\n%s\n%s" % (result.stdout, result.stderr)
    )
    # ``--collect-only -q`` prints one test id per line, each containing
    # the ``::`` separator, followed by a summary line.
    item_ids = [line for line in result.stdout.splitlines() if "::" in line]
    assert item_ids, "pytest collected no tests:\n%s" % result.stdout
    return len(item_ids)


class TestPackageVersion:
    """The distribution metadata must agree with the package."""

    def test_pyproject_version_matches_package_version(self):
        """``pyproject.toml`` and ``dida.__version__`` must agree."""
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r"^version = \"([^\"]+)\"", text, re.MULTILINE)
        assert match is not None, "pyproject.toml does not declare a version."

        assert match.group(1) == dida.__version__, (
            "pyproject.toml declares version %s but dida.__version__ is %s; "
            "the two must not drift apart."
            % (match.group(1), dida.__version__)
        )


class TestDocumentedTestCount:
    """The documented size of the suite must match the real one."""

    @pytest.mark.parametrize("relative_path,pattern", COUNT_CLAIMS)
    def test_documented_count_matches_collected_count(self, relative_path, pattern):
        """Every documented test count must equal the collected count."""
        documented = _documented_count(relative_path, pattern)
        collected = _collected_test_count()

        assert documented == collected, (
            "%s documents %d tests but the suite collects %d; update the "
            "count in every file listed in COUNT_CLAIMS."
            % (relative_path, documented, collected)
        )
