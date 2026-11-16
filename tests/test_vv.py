from validation.run_vv_suite import run_vv_suite


def test_full_vv_suite_has_no_failures():
    """Verification checks must pass; product KPI shortfalls are reported as KNOWN_GAP."""
    report = run_vv_suite(include_ml=False)
    failed = {k: v for k, v in report["tests"].items() if v["status"] == "FAIL"}
    assert not failed, failed
    assert report["summary"] in ("ALL_PASS", "PASS_WITH_KNOWN_GAPS")
