"""
PROTOTYPE for #85, throwaway: dumps every hub, hub grouping, copy number and flank branch eris finds on the three EC958
assemblies in both modes, under the rule changes named in ERIS_P85 (see eris.graph.P85).

Usage: ERIS_P85=hub,entry uv run python scripts/prototype_issue85/dump.py out.json
"""
import json
import sys
from pathlib import Path

import eris.pipeline as pipeline_mod
from eris.constants import FeatureType
from eris.graph import P85
from eris.hub import Hubs
from eris.io import GenomeAssembly, TargetDatabase
from eris.pipeline import Pipeline

DATA = Path("data")
ASSEMBLIES = {"complete": "EC958_complete_genome.fasta", "simulated": "contigs_shovill_simulated.gfa",
              "real": "contigs_shovill_realfastq.gfa"}

log: dict = {}


def record(key, value):
    log.setdefault(key, []).append(value)


# Wrap the hub and branch rules so every decision they make is kept
_of_path, _of_locus, _carried = Hubs.of_path, Hubs.of_locus, pipeline_mod.carried_past_link


def of_path(self, records):
    hub = _of_path(self, records)
    record("path_hubs", [[r.t_name for r in records], sorted(hub.names)])
    return hub


def of_locus(self, locus):
    hub = _of_locus(self, locus)
    record("locus_hubs", [[s.contig for t in locus.targets for s in t.segments], sorted(hub.names)])
    return hub


def carried(link, alignments, lengths):
    n = _carried(link, alignments, lengths)
    record("branches", [f"{link.u}{link.u_strand:+d}>{link.v}{link.v_strand:+d}/{link.overlap}", n])
    return n


Hubs.of_path, Hubs.of_locus, pipeline_mod.carried_past_link = of_path, of_locus, carried


def locus_key(locus):
    return "|".join(f"{s.contig}:{s.start}-{s.end}{s.strand:+d}" for s in locus.targets[0].segments)


def main(out):
    db = TargetDatabase(DATA / "IS.mmi", FeatureType.CDS, indexing_threads=1)
    result = {"P85": sorted(P85)}
    for name, file in ASSEMBLIES.items():
        genome = GenomeAssembly.from_file(DATA / file)
        for mode in ("collapse", "variant"):
            log.clear()
            with Pipeline(db, flank_budget=3, mode=mode, min_coverage=0.30) as p:
                loci = list(p(genome))
            run = {k: sorted(map(json.dumps, v)) for k, v in log.items()}
            run["loci"] = sorted(json.dumps({
                "key": locus_key(l), "target": l.targets[0].id,
                "copy_number": None if l.copy_number is None else
                [l.copy_number.estimate, l.copy_number.edges, l.copy_number.fallback, sorted(l.copy_number.hub)],
                "depth": l.fractional_depth,
                "up": [(f.feature.id, f.distance_bp) for f in l.upstream_flanks],
                "down": [(f.feature.id, f.distance_bp) for f in l.downstream_flanks]}) for l in loci)
            result[f"{name}/{mode}"] = run
            print(name, mode, len(loci), "loci", file=sys.stderr)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
