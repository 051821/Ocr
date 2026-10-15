import pytest

from evals.retrieval_eval import CASES, _stubs, run_case

# Known retrieval gaps found by the eval. strict=True: once one is fixed the test reports XPASS
# as a failure, so remove it from this map.
KNOWN_GAPS = {}


@pytest.fixture(autouse=True)
def _stubbed():
    with _stubs():
        yield


@pytest.mark.parametrize("case", [
    pytest.param(c, id=c.id, marks=pytest.mark.xfail(reason=KNOWN_GAPS[c.id], strict=True))
    if c.id in KNOWN_GAPS else pytest.param(c, id=c.id)
    for c in CASES
])
def test_retrieval_case(case):
    result = run_case(case)
    assert not result.missing, f"evidence missing {result.missing} (path={result.path})"
    assert not result.forbidden_hits, f"evidence contains forbidden {result.forbidden_hits}"
