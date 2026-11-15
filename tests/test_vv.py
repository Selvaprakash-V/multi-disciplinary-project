import pytest
from validation.run_vv_suite import run_vv_suite

def test_full_vv_suite():
    report = run_vv_suite()
    assert report["summary"] == "ALL_PASS"
    assert report["tests"]["dispersion_verification"]["status"] == "PASS"
    assert report["tests"]["energy_cfl_stability"]["status"] == "PASS"

