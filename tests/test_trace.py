from scout.harness.trace import Trace


def test_trace_creates_missing_parent_directories(tmp_path):
    # A fresh checkout (e.g. on Render) has no runs/ directory, since it is git-ignored.
    t = Trace(str(tmp_path / "runs" / "ui"), name="x")
    t.log("order", text="hi")
    assert (tmp_path / "runs" / "ui" / "x.jsonl").read_text().count("\n") == 1
