"""
PROTOTYPE for #85, throwaway: a synthetic graph mixing 0 bp and 127 bp links, the case none of the EC958 assemblies
has. Each case runs under today's rule (global max overlap, summed shortfall) and change 1 (each end against its own
overlap + STRICT_SLACK), and prints both so the difference is visible with `-s`.
"""
import pytest

import eris.hub as hub_mod
from eris.constants import Strand
from eris.graph import Edge
from eris.hub import Hubs
from test_hub import record

F = Strand.FORWARD
LENGTHS = {"is": 1300, "left": 5000, "right": 5000, "far1": 4000, "far2": 4000, "rep": 1300}
EDGES = [
    Edge("is", F, "right", F, 127),   # is: start is a dead end (overlap 0), end links with 127 bp
    Edge("far1", F, "far2", F, 127),  # elsewhere in the graph, so the global max is 127 either way
    Edge("left", F, "rep", F, 127),   # rep: 127 bp links at both ends
    Edge("rep", F, "right", F, 127),
]

CASES = [
    # (what, record, covered today, covered per end)
    ("dead-end start 120 bp unaligned: unique sequence today's rule pulls into the hub",
     record("is", 1300, 120, 1300), True, False),
    ("dead-end start 0 bp short, linked end 120 bp short: within the link's own overlap",
     record("is", 1300, 0, 1180), True, True),
    ("both linked ends 100 bp short: summed shortfall 200 > 137 today, each end within 127 + 10",
     record("rep", 1300, 100, 1200), False, True),
    ("dead-end start 11 bp short: past the strict slack",
     record("is", 1300, 11, 1300), True, False),
]


@pytest.mark.parametrize("what, rec, today, per_end", CASES, ids=[c[0][:40] for c in CASES])
def test_mixed_overlap_coverage(monkeypatch, what, rec, today, per_end):
    hubs = Hubs(EDGES, LENGTHS)
    monkeypatch.setattr(hub_mod, "P85", frozenset())
    got_today = bool(hubs.of_path([rec]))
    monkeypatch.setattr(hub_mod, "P85", frozenset({"hub"}))
    got_per_end = bool(hubs.of_path([rec]))
    print(f"\n{what}: today={got_today} per-end={got_per_end}")
    assert (got_today, got_per_end) == (today, per_end)
