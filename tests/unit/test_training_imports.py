"""Pin the ``vn_news_training`` package import surface.

The first regression we guard against is the upstream mlflow 3.x
circular-import bug seen on Colab: importing ``vn_news_training`` used to
eager-load ``mlflow_utils``, which eager-loaded ``mlflow``, which crashed
during its own class-definition with::

    AttributeError: partially initialized module 'mlflow' has no
    attribute 'version' (most likely due to a circular import)

The fix in ``mlflow_utils.py`` is to lazy-import ``mlflow`` inside each
helper. This test pins that contract by running ``import
vn_news_training`` in a clean subprocess (so that any cached imports in
the test runner don't mask the bug) and then asserting ``mlflow`` is
**not** in ``sys.modules`` afterwards.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import vn_news_training


def test_importing_vn_news_training_does_not_import_mlflow() -> None:
    """Repro for the Colab failure — importing the package must be cheap."""
    script = textwrap.dedent(
        """
        import sys
        # Sanity: mlflow not imported yet.
        assert 'mlflow' not in sys.modules, sorted(k for k in sys.modules if 'mlflow' in k)

        import vn_news_training  # noqa: F401  -- side-effect: load package

        # Real assertion: importing the package must not pull in mlflow.
        leaked = sorted(k for k in sys.modules if k == 'mlflow' or k.startswith('mlflow.'))
        if leaked:
            raise AssertionError(
                f"vn_news_training eagerly imported mlflow modules: {leaked}"
            )
        print('ok')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    msg = f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert result.returncode == 0, msg
    assert result.stdout.strip().endswith("ok")


def test_mlflow_helpers_still_importable() -> None:
    """The helpers themselves must still be importable from the package."""
    # Just sanity-check the public symbols exist; we don't actually
    # invoke them here because that would require an mlflow runtime.
    assert callable(vn_news_training.log_metrics)
    assert callable(vn_news_training.log_params)
    assert callable(vn_news_training.mlflow_run)
