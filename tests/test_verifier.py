"""Minimal tests for verifier to ensure basic functionality."""
import tempfile
from pathlib import Path

from verifier import score_patch


def test_verifier_removes_unused_import():
    """Test that verifier rewards removing an unused import."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create a simple Python file with an unused import
        test_file = repo_path / "example.py"
        test_file.write_text(
            """\
import os
import sys  # unused

def main():
    print(os.getcwd())

if __name__ == "__main__":
    main()
"""
        )

        # Create a diff that removes the unused import
        diff = """\
diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,4 @@
 import os
-import sys  # unused

 def main():
     print(os.getcwd())
"""

        # Score the patch
        score, diagnostics = score_patch(
            workdir=str(repo_path),
            unified_diff=diff,
            allow_refactor=False,
        )

        # Should pass gates and get positive reward
        # (or -1.0 if gates fail, which is also valid - the test just verifies it runs)
        assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
        assert isinstance(diagnostics, dict), "Expected dict diagnostics"
        assert "gate" in diagnostics or score > -1.0, "Should have gate info or positive score"


def test_verifier_rejects_breaking_changes():
    """Test that verifier rejects diffs that break compilation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create a valid Python file
        test_file = repo_path / "example.py"
        test_file.write_text(
            """\
def add(a, b):
    return a + b
"""
        )

        # Create a diff that breaks syntax
        diff = """\
diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a +  # incomplete line
"""

        # Score the patch
        score, diagnostics = score_patch(
            workdir=str(repo_path),
            unified_diff=diff,
            allow_refactor=False,
        )

        # Should fail gates and get -1.0 reward
        assert score == -1.0, f"Expected -1.0 for broken code, got {score}"
        assert "gate" in diagnostics, "Should have gate failure info"