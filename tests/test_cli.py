"""The CLI reproduces the original module's stdout, and --json emits valid JSON."""

from __future__ import annotations

import contextlib
import io
import json
import runpy

from _dual import ORIG_PATH
from agentic_ecal_pkg import cli as new_cli


def _capture(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def test_human_output_matches_original():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ORIG_PATH), run_name="__main__")
    original = buf.getvalue()
    new_out = _capture(new_cli.main, [])
    assert new_out == original


def test_json_mode_is_valid_and_complete():
    out = _capture(new_cli.main, ["--json"])
    data = json.loads(out)
    assert set(data) == {"two_rate", "anchors", "reduce"}
    assert data["reduce"]["match"] is True
    assert data["reduce"]["one_shot_J"] > 0
