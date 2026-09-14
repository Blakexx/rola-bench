import csv

from rola_bench.perf._common import write_csv


def test_write_csv_allows_late_error_fields(tmp_path):
    path = write_csv(tmp_path / "rows.csv", [
        {"bench": "prefill", "arm": "attn", "fwd_ms": 1.0},
        {"bench": "prefill", "arm": "rola", "fwd_ms": "ERR", "error": "boom", "trace": "stack"},
    ])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["fwd_ms"] == "1.0"
    assert rows[1]["error"] == "boom"
    assert rows[1]["trace"] == "stack"
