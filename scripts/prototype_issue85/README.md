# PROTOTYPE #85: contig end reach, measured (throwaway)

The question: does judging each contig end against its own link overlap + a strict slack (10 bp) change hubs, copy
numbers or flank branches on EC958, and does it keep the #52 ratchet?

Toggles: `ERIS_P85=hub` (change 1, hub coverage per end), `ERIS_P85=entry` (change 2, branch entry within the link's
overlap + 10), comma-joined for both. Change 3 (folding `END_SLOP` and `BRANCH_ENTRY_SLACK` into `STRICT_SLACK = 10`)
is a no-op and isn't toggled.

    ERIS_P85=hub uv run python scripts/prototype_issue85/dump.py logs/dump_hub.json   # hubs, groupings, CN, branches
    uv run python scripts/prototype_issue85/diff.py scripts/prototype_issue85/logs    # each config vs base
    ERIS_P85=hub uv run python -m pytest tests/test_integration.py -s                 # ratchets + scores
    uv run python -m pytest tests/test_prototype_issue85_mixed_overlap.py -s          # mixed-overlap fixture

## Results (logs/)

Nothing changes on the complete genome or in variant mode, under any config.

**Change 1 (hub per end):** passes every ratchet. Assignment errors, distance bands, headline F1 and the depth
copy-number F1 are all identical on all three assemblies, in both modes. The global max overlap → per-end change alone
moves nothing on EC958, because no hub contig has a dead end. The summed-shortfall → per-end change adds three
single-contig hubs: sim 1242602, real 2418768 and 2418794. Each is an IS-sized contig with one 111 bp or 71 bp link at
each end, and the IS reaches both ends within overlap + 10 (gaps 118/103 on sim 1242602; summed 221 > 121 today).
Their copy number stays 1 by depth but falls back (edges 1 → 0). The walks out of the IS600 and ISEc9 hubs next to them
now stop at them as repeats, so those hubs lose an edge each (2 → 1, 3 → 2). As a result the `copy_number_edges` isoform
hub F1 falls: simulated 0.929 → 0.907, real 0.898 → 0.878. That column is printed for comparison (#47), not asserted.

**Change 2 (branch entry within overlap + 10):** fails the real-collapse distance-band ratchet with a new
disagreement, (IS1A, downstream, 2420448_1187_1312), and its F1 falls 0.777 → 0.773. Cause: branch 2420860+ → 2420448-
(71 bp overlap) now carries 107 bp from an **ISKpn37** alignment (IS3 family) that starts at walk position 71, exactly
where the overlap ends. That alignment is a neighbouring IS, not the IS1A copy. A copy carrying on past a hub edge
always starts at ~0, because the overlap is shared sequence. Verdict: fall back to the slack-only check (start ≤ 10),
which is today's rule.

**Fixture** (`tests/test_prototype_issue85_mixed_overlap.py`): a dead-end start 120 bp unaligned is covered today and
not per end. Two linked ends each 100 bp short are not covered today (summed 200 > 137) but are covered per end.
