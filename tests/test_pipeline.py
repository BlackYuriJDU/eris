from types import SimpleNamespace

import pytest
import numpy as np
from Bio.Seq import Seq

from eris.alignment import AlignmentBatch
from eris.constants import FeatureType, Strand
from eris.graph import Edge, TopologyEngine
from eris.interval import IntervalBatch
from eris.io import GenomeAssembly, format_contig_path
from eris.pipeline import LocusBuilder, Locus, GenomicFeature, LocationSegment


def _batch(rows, strands=None):
    """rows: (q_name, q_length, q_start, q_end, t_name, t_length, t_start, t_end, length, match); all '+' by default"""
    qn, ql, qs, qe, tn, tl, ts, te, bl, ml = zip(*rows)
    return AlignmentBatch(
        q_names=np.array(qn, dtype=object), q_lengths=np.array(ql, dtype=np.int32),
        q_starts=np.array(qs, dtype=np.int32), q_ends=np.array(qe, dtype=np.int32),
        t_names=np.array(tn, dtype=object), t_lengths=np.array(tl, dtype=np.int32),
        t_starts=np.array(ts, dtype=np.int32), t_ends=np.array(te, dtype=np.int32),
        strands=np.array(strands or [Strand.FORWARD] * len(rows), dtype=np.int8),
        lengths=np.array(bl, dtype=np.int32), matches=np.array(ml, dtype=np.int32),
        mismatches=np.array([b - m for b, m in zip(bl, ml)], dtype=np.int32),
        qualities=np.array([60] * len(rows), dtype=np.int32),
        cigars=np.array([f"{b}M" for b in bl], dtype=object)
    )


def _builder(lengths, edges=(), mode='variant', depths=None, intervals=None, max_feature_hops=0, circular=frozenset(),
             **builder_args):
    depths = depths if depths is not None else {c: 10.0 for c in lengths}
    genome = GenomeAssembly("test", {c: Seq("A" * n) for c, n in lengths.items()}, list(edges), depths, lengths,
                            frozenset(circular))
    engine = TopologyEngine(list(edges), lengths, depths, features=intervals, mode=mode, circular=circular)
    return LocusBuilder(engine, genome, max_feature_hops=max_feature_hops, **builder_args)


def test_assemble_drops_local_alignments_below_min_identity():
    builder = _builder({"ctgA": 10000})
    alignments = {"ctgA": _batch([
        ("IS1", 1000, 0, 1000, "ctgA", 10000, 1000, 2000, 1000, 990),
        ("IS1", 1000, 0, 1000, "ctgA", 10000, 6000, 7000, 1000, 800),
    ])}

    loci = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.90))

    assert [(locus.start, [t.identity for t in locus.targets]) for locus in loci] == [(1000, [pytest.approx(0.99)])]


def test_assemble_pools_identity_of_a_stitched_locus_over_its_real_fragments():
    # IS1 runs off the end of ctgA, across the unaligned ctgB, and into the start of ctgC
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
             Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)]
    builder = _builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000}, edges)
    alignments = {
        "ctgA": _batch([("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000, 1000, 950)]),
        "ctgC": _batch([("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500, 500, 400)]),
    }

    # (950 + 400) / (1000 + 500) = 0.90; counting ctgB's 500 bp would give 0.675
    kept = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.85))
    assert [t.identity for locus in kept for t in locus.targets] == [pytest.approx(0.90)]

    assert list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.95)) == []


# Adjacent IS copies (#57) ------------------------------------------------------------------------------------------
def test_two_abutting_copies_of_different_iss_are_two_loci_each_with_its_own_strand_and_sides():
    # ISEc9 on '+' at 1000-2000 and IS5 on '-' at 1998-3200 share 2 bp, as in EC958. Each is the other's flank on
    # the shared side: ISEc9's downstream and IS5's downstream (IS5 is read backwards, so its downstream is leftwards)
    genes = [(100, 900), (1100, 1900), (2100, 3100), (3300, 4000)]  # left flank, ISEc9 tnp, IS5 tnp, right flank
    intervals = {"ctgA": IntervalBatch(np.array([s for s, _ in genes]), np.array([e for _, e in genes]),
                                       np.array([Strand.FORWARD] * len(genes), dtype=np.int8))}
    features = {"ctgA": [GenomicFeature(f"g{s}", FeatureType.CDS, [LocationSegment("ctgA", s, e, Strand.FORWARD)])
                         for s, e in genes]}
    no_indels = {"ctgA": [SimpleNamespace(insertions=[], deletions=[]) for _ in genes]}
    builder = _builder({"ctgA": 5000}, intervals=intervals, max_feature_hops=1, features=features, genes=no_indels)
    alignments = {"ctgA": _batch([("ISEc9", 1000, 0, 1000, "ctgA", 5000, 1000, 2000, 1000, 1000),
                                  ("IS5", 1202, 0, 1202, "ctgA", 5000, 1998, 3200, 1202, 1202)],
                                 strands=[Strand.FORWARD, Strand.REVERSE])}

    loci = sorted(builder.assemble(alignments), key=lambda locus: locus.start)

    def sides(locus):
        return ([f.feature.id for f in locus.upstream_flanks], [f.feature.id for f in locus.downstream_flanks])
    assert [([t.id for t in locus.targets], locus.start, locus.end) for locus in loci] == [
        (["ISEc9"], 1000, 2000), (["IS5"], 1998, 3200)]
    assert [locus.targets[0].bounding_strand for locus in loci] == [Strand.FORWARD, Strand.REVERSE]
    assert [sides(locus) for locus in loci] == [(["g100"], ["g2100"]), (["g3300"], ["g1100"])]
    assert [[p.feature.id for p in locus.passengers] for locus in loci] == [["g1100"], ["g2100"]]


def test_two_near_identical_targets_on_one_copy_are_one_locus():
    # IS1A and IS1R align to the same 768 bp copy, a few bases apart at the ends
    builder = _builder({"ctgA": 5000})
    alignments = {"ctgA": _batch([("IS1A", 768, 0, 768, "ctgA", 5000, 1000, 1768, 768, 760),
                                  ("IS1R", 768, 0, 766, "ctgA", 5000, 1002, 1768, 766, 750)])}

    [locus] = builder.assemble(alignments)

    assert (sorted(t.id for t in locus.targets), locus.start, locus.end) == (["IS1A", "IS1R"], 1000, 1768)


def test_two_abutting_copies_of_the_same_is_are_two_loci():
    builder = _builder({"ctgA": 5000})
    alignments = {"ctgA": _batch([("IS26", 820, 0, 820, "ctgA", 5000, 1000, 1820, 820, 820),
                                  ("IS26", 820, 0, 820, "ctgA", 5000, 1820, 2640, 820, 820)])}

    loci = builder.assemble(alignments)

    assert sorted((locus.start, locus.end) for locus in loci) == [(1000, 1820), (1820, 2640)]


@pytest.mark.parametrize("strand, query_ranges", [(Strand.FORWARD, [(0, 500), (520, 1000)]),
                                                  (Strand.REVERSE, [(520, 1000), (0, 500)])])
def test_one_copy_split_into_two_touching_alignments_of_its_target_is_one_locus(strand, query_ranges):
    # A deletion in the contig splits IS1's alignment in two: the pieces touch on the contig and follow on in IS1
    builder = _builder({"ctgA": 5000})
    (q1s, q1e), (q2s, q2e) = query_ranges
    alignments = {"ctgA": _batch([("IS1", 1000, q1s, q1e, "ctgA", 5000, 1000, 1500, 500, 500),
                                  ("IS1", 1000, q2s, q2e, "ctgA", 5000, 1500, 1980, 480, 480)], strands=[strand] * 2)}

    [locus] = builder.assemble(alignments)

    assert (locus.start, locus.end) == (1000, 1980)


@pytest.mark.parametrize("second_start, expected_loci", [(1500, 1), (1501, 2)])
def test_alignments_share_a_locus_only_when_they_share_at_least_half_the_shorter_one(second_start, expected_loci):
    # The shorter alignment is 1000 bp: sharing 500 of its bases is one locus, 499 is two
    builder = _builder({"ctgA": 5000})
    alignments = {"ctgA": _batch([("ISa", 1000, 0, 1000, "ctgA", 5000, 1000, 2000, 1000, 1000),
                                  ("ISb", 1200, 0, 1200, "ctgA", 5000, second_start, second_start + 1200, 1200, 1200)])}

    assert len(list(builder.assemble(alignments))) == expected_loci


def test_stitched_locus_reports_the_strand_each_contig_is_read_on():
    # IS1 runs off ctgA, through the unaligned ctgB read backwards ('ctgA+ -> ctgB-'), and into ctgC
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=0),
             Edge("ctgB", Strand.REVERSE, "ctgC", Strand.FORWARD, overlap=0)]
    builder = _builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000}, edges)
    alignments = {
        "ctgA": _batch([("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000, 1000, 1000)]),
        "ctgC": _batch([("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500, 500, 500)]),
    }

    loci = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.0))

    assert [format_contig_path(locus) for locus in loci] == ["ctgA+,ctgB-,ctgC+"]


def _two_targets_through_one_hub(mode):
    """
    IS1A and IS1B are near-identical: each runs off its own anchor, across the unaligned 1 kb hub, and into another
    anchor. IS1A aligns 600 bp on its anchors, IS1B only 200 bp.
    """
    hub, anchors = {"hub": 1000}, {c: 5000 for c in ("a1", "b1", "a2", "b2")}
    edges = [Edge("a1", Strand.FORWARD, "hub", Strand.FORWARD), Edge("hub", Strand.FORWARD, "b1", Strand.FORWARD),
             Edge("a2", Strand.FORWARD, "hub", Strand.FORWARD), Edge("hub", Strand.FORWARD, "b2", Strand.FORWARD)]
    builder = _builder(hub | anchors, edges, mode=mode, depths={"hub": 20.0} | {c: 10.0 for c in anchors})
    alignments = {
        "a1": _batch([("IS1A", 1600, 0, 300, "a1", 5000, 4700, 5000, 300, 300)]),
        "b1": _batch([("IS1A", 1600, 1300, 1600, "b1", 5000, 0, 300, 300, 300)]),
        "a2": _batch([("IS1B", 1400, 0, 100, "a2", 5000, 4900, 5000, 100, 100)]),
        "b2": _batch([("IS1B", 1400, 1300, 1400, "b2", 5000, 0, 100, 100, 100)]),
    }
    return list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.0))


def test_collapse_mode_reports_a_hub_once_when_near_identical_targets_both_stitch_through_it():
    # The hub is reported once, named for IS1A, which aligns more of itself
    loci = _two_targets_through_one_hub('collapse')

    assert [(t.id, [s.contig for s in t.segments]) for locus in loci for t in locus.targets] == [
        ("IS1A_stitched", ["a1", "hub", "b1"])]


def test_variant_mode_keeps_every_stitched_path_through_a_hub():
    loci = _two_targets_through_one_hub('variant')

    assert sorted(t.id for locus in loci for t in locus.targets) == ["IS1A_stitched", "IS1B_stitched"]


def _path_locus(*segments):
    """A locus whose one target runs along segments of (contig, start, end, strand), in the target's direction."""
    target = GenomicFeature("IS1", FeatureType.CDS, [LocationSegment(*s) for s in segments])
    contig = "|".join(s[0] for s in segments)
    return Locus("locus_1", contig, segments[0][1], segments[-1][2], [target], [], [], [])


def _genome(contigs, edges):
    return GenomeAssembly("test", {c: Seq(s) for c, s in contigs.items()}, edges, {}, {c: len(s) for c, s in contigs.items()})


# ctgB read backwards (TTTACGTGGG reversed: GGGTTTACGT) starts with the last 3 bp of ctgA, the edge overlap
_A, _B = "AAAACCCGGG", "ACGTAAACCC"


def test_stitched_sequence_joins_contigs_on_their_path_strands_counting_each_overlap_once():
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 0, 10, Strand.FORWARD), ("ctgB", 0, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AAAACCCGGG" + "TTTACGT"


def test_stitched_sequence_is_cut_to_the_target_ends_on_the_first_and_last_contigs():
    # The target starts 4 bp into ctgA and ends 7 bp into ctgB as it is read (ctgB 3..10 on its forward strand)
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 4, 10, Strand.FORWARD), ("ctgB", 3, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "CCCGGG" + "TTTA"


def test_stitched_sequence_finds_the_overlap_of_a_link_written_on_the_other_strands():
    # 'ctgB+ -> ctgA-' is the same link as 'ctgA+ -> ctgB-'
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgB", Strand.FORWARD, "ctgA", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 0, 10, Strand.FORWARD), ("ctgB", 0, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AAAACCCGGG" + "TTTACGT"


def test_stitched_sequence_read_backwards_starts_from_the_target_start_on_the_reverse_strand():
    # The same target walked the other way, 'ctgB+ -> ctgA-'
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgB", 3, 10, Strand.FORWARD), ("ctgA", 4, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == str(Seq("CCCGGG" + "TTTA").reverse_complement())


def test_stitched_sequence_keeps_an_unaligned_intermediate_contig_whole_but_for_its_overlap():
    # 'ctgA+ -> ctgB- -> ctgC+', each link overlapping by 2 bp; the target claims all of the unaligned ctgB
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=2),
             Edge("ctgB", Strand.REVERSE, "ctgC", Strand.FORWARD, overlap=2)]
    genome = _genome({"ctgA": "GGGGAC", "ctgB": "TTTGTC", "ctgC": "AAACCCC"}, edges)
    locus = _path_locus(("ctgA", 3, 6, Strand.FORWARD), ("ctgB", 0, 6, Strand.REVERSE), ("ctgC", 0, 4, Strand.FORWARD))

    assert locus.extract_sequence(genome) == "GAC" + "CAAA" + "AC"


def test_local_sequence_is_its_slice_of_the_contig():
    genome = _genome({"ctgA": _A}, [])
    locus = _path_locus(("ctgA", 2, 8, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AACCCG"


def _flank_builder(lengths, edges, gene_spans, mode='variant', **builder_args):
    """gene_spans: contig -> (start, end) of its one gene, read on '+'; flanks are searched one gene deep."""
    intervals = {c: IntervalBatch(np.array([s]), np.array([e]), np.array([Strand.FORWARD], dtype=np.int8))
                 for c, (s, e) in gene_spans.items()}
    features = {c: [GenomicFeature(f"gene_{c}", FeatureType.CDS, [LocationSegment(c, s, e, Strand.FORWARD)])]
                for c, (s, e) in gene_spans.items()}
    no_indels = {c: [SimpleNamespace(insertions=[], deletions=[])] for c in gene_spans}
    return _builder(lengths, edges, mode=mode, intervals=intervals, max_feature_hops=1, features=features,
                    genes=no_indels, **builder_args)


def test_downstream_flank_across_a_link_counts_the_overlap_once():
    # 'ctgA+ -> ctgB+' share 100 bp, so ctgB starts at ctgA's 900 and its gene at 300 sits at 1200 in the joined sequence
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (300, 600)})
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [f.distance_bp for f in locus.downstream_flanks] == [200]


def test_upstream_flank_across_a_link_entered_backwards_counts_the_overlap_once():
    # 'ctgB+ -> ctgA+' share 100 bp; walking upstream leaves ctgA by its start and reads ctgB on '-'. ctgA starts at
    # ctgB's 900, so the gene ending at 600 on ctgB is 300 bp before the locus at ctgA's start
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgB", Strand.FORWARD, "ctgA", Strand.FORWARD, overlap=100)],
                             {"ctgB": (300, 600)})
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 0, 200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [f.distance_bp for f in locus.upstream_flanks] == [300]


def test_flank_two_links_away_counts_each_overlap_once():
    # 'ctgA+ -> ctgB+ -> ctgC+' share 100 and 50 bp; ctgB carries no gene. ctgB starts at ctgA's 900 and ctgC at
    # 900 + 500 - 50 = 1350, so ctgC's gene at 200 sits at 1550, 550 bp past the locus end
    builder = _flank_builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100),
                              Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=50)],
                             {"ctgC": (200, 400)})
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [(f.distance_bp, f.topological_dist) for f in locus.downstream_flanks] == [(550, 2)]


def _relation_summary(relations):
    return [(r.feature.id, r.spatial.name, r.distance_bp, r.effect.name) for r in relations]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("strand, aln_start, side, context", [
    # A '+' IS ends at ctgA's end; downstream is across the link
    (Strand.FORWARD, 800, "downstream", "OVERLAPPING_END"),
    # A '-' IS reads ctgA backwards, so the link past ctgA's end is upstream of it
    (Strand.REVERSE, 800, "upstream", "OVERLAPPING_START"),
])
def test_a_gene_on_the_next_contig_that_runs_back_into_the_is_end_overlaps_it(mode, strand, aln_start, side, context):
    # 'ctgA+ -> ctgB+' share 100 bp, so ctgB's gene at 30 sits at ctgA's 930, 70 bp inside the IS ending at 1000
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (30, 600)}, mode=mode)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, aln_start, aln_start + 200, 200, 200)],
                                 [strand])}

    [locus] = builder.assemble(alignments)

    assert _relation_summary(getattr(locus, f"{side}_flanks")) == [("gene_ctgB", context, 0, "DISRUPTED")]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("strand, side, context", [
    (Strand.FORWARD, "upstream", "OVERLAPPING_START"),
    (Strand.REVERSE, "downstream", "OVERLAPPING_END"),
])
def test_a_gene_on_the_previous_contig_that_runs_into_the_is_start_overlaps_it(mode, strand, side, context):
    # 'ctgB+ -> ctgA+' share 100 bp, so ctgA starts at ctgB's 900 and ctgB's gene ending at 970 runs 70 bp into the IS
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgB", Strand.FORWARD, "ctgA", Strand.FORWARD, overlap=100)],
                             {"ctgB": (300, 970)}, mode=mode)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 0, 200, 200, 200)], [strand])}

    [locus] = builder.assemble(alignments)

    assert _relation_summary(getattr(locus, f"{side}_flanks")) == [("gene_ctgB", context, 0, "DISRUPTED")]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("strand, aln, side", [
    # A '+' IS aligns its first 300 of 500 bp up to 5 bp short of ctgA's end, so its last 200 bp run on into ctgB
    (Strand.FORWARD, (0, 300), "downstream"),
    # Read backwards, its last 300 bp end there, so its first 200 bp run on into ctgB, upstream of it
    (Strand.REVERSE, (200, 500), "upstream"),
])
def test_a_flank_across_a_link_is_measured_from_where_the_target_ends_when_its_alignment_runs_off_the_contig(
        mode, strand, aln, side):
    # ctgB starts at ctgA's 900, so its gene at 500 sits at 1400, 405 bp past the alignment's end at 995: 205 bp past
    # the target's end 200 bp further on
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (500, 800)}, mode=mode)
    alignments = {"ctgA": _batch([("IS1", 500, *aln, "ctgA", 1000, 695, 995, 300, 300)], [strand])}

    [locus] = builder.assemble(alignments)

    assert [f.distance_bp for f in getattr(locus, f"{side}_flanks")] == [205]


def test_a_flank_across_a_link_is_measured_from_the_alignment_end_when_it_stops_inside_the_contig():
    # The alignment stops 200 bp short of ctgA's end, more than the link's overlap and slack: that copy is truncated,
    # so its missing 200 bp are not on ctgB, and ctgB's gene at 1400 is 600 bp past the alignment's end at 800
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (500, 800)})
    alignments = {"ctgA": _batch([("IS1", 500, 0, 300, "ctgA", 1000, 500, 800, 300, 300)])}

    [locus] = builder.assemble(alignments)

    assert [f.distance_bp for f in locus.downstream_flanks] == [600]


def test_a_flank_across_a_link_is_measured_from_the_alignment_end_when_the_rest_of_the_target_would_fit_on_its_contig():
    # The alignment stops 120 bp short of ctgA's end, within the link's overlap and slack, but leaves only 20 bp of
    # the target unaligned: they would lie on ctgA, where they do not align, so the copy is truncated and ctgB's gene
    # at 1400 is 520 bp past the alignment's end at 880
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (500, 800)})
    alignments = {"ctgA": _batch([("IS1", 500, 0, 480, "ctgA", 1000, 400, 880, 480, 480)])}

    [locus] = builder.assemble(alignments)

    assert [f.distance_bp for f in locus.downstream_flanks] == [520]


def test_a_stitched_path_whose_first_alignment_runs_off_its_contig_is_measured_from_the_targets_start():
    # IS1 starts on ctgX at its position 200: its first 200 bp run back onto ctgU, whose gene ending at 700 is 300 bp
    # before ctgX starts, so 100 bp before the target does
    edges = [Edge("ctgU", Strand.FORWARD, "ctgX", Strand.FORWARD), Edge("ctgX", Strand.FORWARD, "ctgY", Strand.FORWARD)]
    builder = _flank_builder({"ctgU": 1000, "ctgX": 300, "ctgY": 1000}, edges, {"ctgU": (300, 700)})
    alignments = {
        "ctgX": _batch([("IS1", 1000, 200, 500, "ctgX", 300, 0, 300, 300, 300)]),
        "ctgY": _batch([("IS1", 1000, 500, 1000, "ctgY", 1000, 0, 500, 500, 500)]),
    }

    [locus] = builder.assemble(alignments)

    assert format_contig_path(locus) == "ctgX+,ctgY+"
    assert [f.distance_bp for f in locus.upstream_flanks] == [100]


def test_a_stitched_path_read_backwards_whose_first_alignment_runs_off_its_contig_is_measured_from_the_targets_start():
    # As above on '-': IS1 is read along ctgX- then ctgY-, so its first 200 bp run off ctgX's end onto ctgU, whose gene
    # starting at 300 is 300 bp past ctgX's end, so 100 bp before the target's start
    edges = [Edge("ctgY", Strand.FORWARD, "ctgX", Strand.FORWARD), Edge("ctgX", Strand.FORWARD, "ctgU", Strand.FORWARD)]
    builder = _flank_builder({"ctgU": 1000, "ctgX": 300, "ctgY": 1000}, edges, {"ctgU": (300, 700)})
    alignments = {
        "ctgX": _batch([("IS1", 1000, 200, 500, "ctgX", 300, 0, 300, 300, 300)], [Strand.REVERSE]),
        "ctgY": _batch([("IS1", 1000, 500, 1000, "ctgY", 1000, 500, 1000, 500, 500)], [Strand.REVERSE]),
    }

    [locus] = builder.assemble(alignments)

    assert format_contig_path(locus) == "ctgX-,ctgY-"
    assert [f.distance_bp for f in locus.upstream_flanks] == [100]


def test_a_gene_across_a_link_is_not_truncated_by_a_frameshift_at_the_is_coordinates_on_its_own_contig():
    # ctgB's gene has a frameshift at 805, which is near the IS start, 800, but on ctgA, not ctgB
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgB": (300, 900)})
    builder.genes["ctgB"] = [SimpleNamespace(insertions=[805], deletions=[],
                                             mutations=lambda seq: [SimpleNamespace(pos=805)])]
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _relation_summary(locus.downstream_flanks) == [("gene_ctgB", "DOWNSTREAM", 200, "NONE")]


@pytest.mark.parametrize("strand, gene, context", [
    (Strand.FORWARD, (300, 450), "OVERLAPPING_START"),
    (Strand.FORWARD, (550, 800), "OVERLAPPING_END"),
    (Strand.REVERSE, (550, 800), "OVERLAPPING_START"),
    (Strand.REVERSE, (300, 450), "OVERLAPPING_END"),
])
def test_a_local_gene_overlapping_the_is_upstream_end_overlaps_its_start_and_downstream_end_its_end(strand, gene, context):
    # The graph walk follows the same convention: START is the upstream end in the IS's own direction
    builder = _flank_builder({"ctgA": 10000}, [], {"ctgA": gene})
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 10000, 400, 600, 200, 200)], [strand])}

    [locus] = builder.assemble(alignments)

    assert _relation_summary(locus.passengers) == [("gene_ctgA", context, 0, "DISRUPTED")]


@pytest.mark.parametrize("limit, reported", [(499, False), (500, True)])
@pytest.mark.parametrize("gene, aln_start, side", [((1500, 1800), 800, "downstream"), ((100, 300), 800, "upstream")])
def test_a_flank_on_the_targets_own_contig_is_reported_only_within_the_max_flank_distance(
        limit, reported, gene, aln_start, side):
    # The gene is 500 bp from the IS on either side
    builder = _flank_builder({"ctgA": 10000}, [], {"ctgA": gene}, max_flank_distance=limit)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 10000, aln_start, aln_start + 200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(getattr(locus, f"{side}_flanks")) == ([("gene_ctgA", 500)] if reported else [])


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("limit, reported", [(549, False), (550, True)])
def test_a_flank_across_links_is_reported_only_within_the_max_flank_distance(mode, limit, reported):
    # ctgB carries no gene, so --hops alone does not stop the walk; ctgC's gene is 550 bp past the locus end
    builder = _flank_builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100),
                              Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=50)],
                             {"ctgC": (200, 400)}, mode=mode, max_flank_distance=limit)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.downstream_flanks) == ([("gene_ctgC", 550)] if reported else [])


@pytest.mark.parametrize("limit, reported", [(299, False), (300, True)])
def test_an_upstream_flank_across_a_link_entered_backwards_is_reported_only_within_the_max_flank_distance(
        limit, reported):
    # 'ctgB+ -> ctgA+' share 100 bp, so the gene ending at 600 on ctgB is 300 bp before the locus at ctgA's start
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgB", Strand.FORWARD, "ctgA", Strand.FORWARD, overlap=100)],
                             {"ctgB": (300, 600)}, max_flank_distance=limit)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 0, 200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == ([("gene_ctgB", 300)] if reported else [])


def test_a_gene_beyond_the_max_flank_distance_does_not_use_up_the_flank_budget():
    # Two branches leave ctgA: the gene on ctgFar is beyond the limit, so the one on ctgNear still gets reported
    builder = _flank_builder({"ctgA": 1000, "ctgFar": 5000, "ctgNear": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgFar", Strand.FORWARD),
                              Edge("ctgA", Strand.FORWARD, "ctgNear", Strand.FORWARD)],
                             {"ctgFar": (4000, 4300), "ctgNear": (100, 400)}, max_flank_distance=1000)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.downstream_flanks) == [("gene_ctgNear", 100)]


@pytest.mark.parametrize("order", [("ctgA", "ctgB"), ("ctgB", "ctgA")])
def test_two_loci_whose_walks_reach_the_same_gene_both_report_it(order):
    # 'ctgA+ -> ctgC+' and 'ctgB+ -> ctgC+': each IS runs off its contig's end into ctgC, so ctgC's gene flanks both
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000, "ctgC": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgC", Strand.FORWARD),
                              Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD)],
                             {"ctgC": (300, 600)})
    alignments = {c: _batch([("IS1", 200, 0, 200, c, 1000, 800, 1000, 200, 200)]) for c in order}

    loci = list(builder.assemble(alignments))

    assert sorted((locus.contig, _flank_genes(locus.downstream_flanks)) for locus in loci) == [
        ("ctgA", [("gene_ctgC", 300)]), ("ctgB", [("gene_ctgC", 300)])]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_a_gene_two_branches_of_one_locus_reach_is_reported_once(mode):
    # 'ctgA+ -> ctgB+' and 'ctgA+ -> ctgB-' enter ctgB on both strands, and both reach its one gene
    builder = _flank_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD),
                              Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE)],
                             {"ctgB": (300, 600)}, mode=mode)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [f.feature.id for f in locus.downstream_flanks] == ["gene_ctgB"]


def _circular_builder(length, gene_spans, overlap=0, hops=1, **builder_args):
    """One circular contig 'chrom', closed by a self-link, carrying genes read on '+' at gene_spans."""
    starts, ends = zip(*gene_spans)
    intervals = {"chrom": IntervalBatch(np.array(starts), np.array(ends),
                                        np.array([Strand.FORWARD] * len(gene_spans), dtype=np.int8))}
    features = {"chrom": [GenomicFeature(f"gene_{s}", FeatureType.CDS, [LocationSegment("chrom", s, e, Strand.FORWARD)])
                          for s, e in gene_spans]}
    no_indels = {"chrom": [SimpleNamespace(insertions=[], deletions=[]) for _ in gene_spans]}
    return _builder({"chrom": length}, [Edge("chrom", Strand.FORWARD, "chrom", Strand.FORWARD, overlap=overlap)],
                    intervals=intervals, max_feature_hops=hops, features=features, genes=no_indels,
                    circular={"chrom"}, **builder_args)


def _flank_genes(flanks):
    return [(f.feature.id, f.distance_bp) for f in flanks]


def test_a_gene_just_past_the_origin_flanks_a_target_at_the_end_of_a_circular_contig():
    builder = _circular_builder(10000, [(300, 600), (5000, 5500)])
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 10000, 9800, 10000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.downstream_flanks) == [("gene_300", 300)]
    assert _flank_genes(locus.upstream_flanks) == [("gene_5000", 4300)]


def test_a_gene_just_before_the_origin_flanks_a_target_at_the_start_of_a_circular_contig():
    builder = _circular_builder(10000, [(300, 600), (5000, 5500)])
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 10000, 0, 200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == [("gene_5000", 4500)]
    assert _flank_genes(locus.downstream_flanks) == [("gene_300", 100)]


@pytest.mark.parametrize("limit, reported", [(299, False), (300, True)])
def test_a_flank_across_the_origin_is_reported_only_within_the_max_flank_distance(limit, reported):
    builder = _circular_builder(10000, [(300, 600), (5000, 5500)], max_flank_distance=limit)
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 10000, 9800, 10000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.downstream_flanks) == ([("gene_300", 300)] if reported else [])


def test_a_flank_across_the_origin_counts_the_self_link_overlap_once():
    # The contig's last 100 bp repeat its first 100 bp, so the gene at 300 sits 200 bp past the contig's end
    builder = _circular_builder(10000, [(300, 600), (5000, 5500)], overlap=100)
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 10000, 9800, 10000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.downstream_flanks) == [("gene_300", 200)]


def test_a_walk_across_the_origin_stops_before_the_target_it_started_from():
    # The only gene carries the target, so going round the circle finds no gene on either side of it
    builder = _circular_builder(3000, [(1000, 2000)], hops=3)
    alignments = {"chrom": _batch([("IS1", 300, 0, 300, "chrom", 3000, 1200, 1500, 300, 300)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.passengers) == [("gene_1000", 0)]
    assert locus.upstream_flanks == locus.downstream_flanks == []


def test_a_self_link_beside_other_links_is_not_walked_as_an_origin():
    # 'rep' is a collapsed tandem repeat: its self-link is not a way back to its own start
    intervals = {"rep": IntervalBatch(np.array([100]), np.array([400]), np.array([Strand.FORWARD], dtype=np.int8))}
    features = {"rep": [GenomicFeature("gene_rep", FeatureType.CDS,
                                       [LocationSegment("rep", 100, 400, Strand.FORWARD)])]}
    no_indels = {"rep": [SimpleNamespace(insertions=[], deletions=[])]}
    edges = [Edge("rep", Strand.FORWARD, "rep", Strand.FORWARD), Edge("ctgA", Strand.FORWARD, "rep", Strand.FORWARD)]
    builder = _builder({"rep": 2000, "ctgA": 5000}, edges, intervals=intervals, max_feature_hops=1,
                       features=features, genes=no_indels)
    alignments = {"rep": _batch([("IS1", 200, 0, 200, "rep", 2000, 1800, 2000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert locus.downstream_flanks == []


def _hub_locus(mode, extra_edges=(), extra_genes=None, extra_alignments=None):
    """
    IS1 runs off the end of a1, across the unaligned 1 kb 'hub', and into the start of b1. The hub is a collapsed
    repeat: a2 also links into it, and b2 (sharing 100 bp) and b3 (entered backwards) also leave it.
    """
    lengths = {"hub": 1000} | {c: 5000 for c in ("a1", "a2")} | {c: 1000 for c in ("b1", "b2", "b3")}
    edges = [Edge("a1", Strand.FORWARD, "hub", Strand.FORWARD), Edge("a2", Strand.FORWARD, "hub", Strand.FORWARD),
             Edge("hub", Strand.FORWARD, "b1", Strand.FORWARD),
             Edge("hub", Strand.FORWARD, "b2", Strand.FORWARD, overlap=100),
             Edge("hub", Strand.FORWARD, "b3", Strand.REVERSE), *extra_edges]
    genes = {"a1": (4000, 4500), "a2": (4200, 4600), "b1": (500, 800), "b2": (300, 600), "b3": (600, 750)}
    builder = _flank_builder(lengths, edges, genes | (extra_genes or {}), mode=mode,
                             depths={"hub": 30.0} | {c: 10.0 for c in lengths if c != "hub"})
    alignments = {
        "a1": _batch([("IS1", 1600, 0, 300, "a1", 5000, 4700, 5000, 300, 300)]),
        "b1": _batch([("IS1", 1600, 1300, 1600, "b1", 1000, 0, 300, 300, 300)]),
    } | (extra_alignments or {})
    [locus] = builder.assemble(alignments, min_coverage=0.30, min_identity=0.0)
    return locus


def _flank_relations(flanks):
    return sorted((f.feature.id, f.distance_bp, f.topological_dist) for f in flanks)


def test_collapse_mode_walks_a_hubs_flanks_from_every_link_leaving_it():
    # b1 carries the IS's end, so its gene is 200 bp from the IS on the contig itself. The other branches are measured
    # from the hub's edge: b2's gene at 300 is 200 bp past the 100 bp it shares with the hub, and b3, read backwards,
    # starts with its gene ending at 750, 250 bp in. Upstream, a2's gene ends 400 bp before the hub
    locus = _hub_locus('collapse')

    assert _flank_relations(locus.downstream_flanks) == [
        ("gene_b1", 200, 0), ("gene_b2", 200, 1), ("gene_b3", 250, 1)]
    assert _flank_relations(locus.upstream_flanks) == [("gene_a1", 200, 0), ("gene_a2", 400, 1)]


def test_a_hub_branch_is_measured_from_where_its_own_copys_target_ends():
    # IS1B, near-identical to IS1, is another copy's target, aligned on one contig each so it stitches no path of its
    # own. It runs past the hub onto b2 and b3: on b2 to 200, 100 bp past the overlap, so its gene at
    # 300 is 100 bp away; on b3, read backwards, its last 150 bp, so the gene starting 250 bp in is 100 bp away.
    # Upstream, a2 ends with its copy's first 200 bp, 200 bp short of the gene's end at 4600
    alignments = {
        "b2": _batch([("IS1B", 1600, 1400, 1600, "b2", 1000, 0, 200, 200, 200)]),
        "b3": _batch([("IS1B", 1600, 1450, 1600, "b3", 1000, 850, 1000, 150, 150)], [Strand.REVERSE]),
        "a2": _batch([("IS1B", 1600, 0, 200, "a2", 5000, 4800, 5000, 200, 200)]),
    }
    locus = _hub_locus('collapse', extra_alignments=alignments)

    assert _flank_relations(locus.downstream_flanks) == [
        ("gene_b1", 200, 0), ("gene_b2", 100, 1), ("gene_b3", 100, 1)]
    assert _flank_relations(locus.upstream_flanks) == [("gene_a1", 200, 0), ("gene_a2", 200, 1)]


def test_a_hub_branch_ignores_an_alignment_that_starts_past_the_overlap():
    # IS3X starts on b2 just where the 100 bp it shares with the hub ends: it is another IS beside this copy, not this
    # copy's target running on past the hub, so this copy's target still ends where b2 is entered
    alignments = {"b2": _batch([("IS3X", 1200, 1050, 1200, "b2", 1000, 100, 250, 150, 150)])}
    locus = _hub_locus('collapse', extra_alignments=alignments)

    assert ("gene_b2", 200, 1) in _flank_relations(locus.downstream_flanks)


def test_links_leading_back_into_the_hub_are_not_branches_of_its_flank_search():
    # The hairpin 'hub+ -> hub-' leaves the hub's end but turns back into it: walking it would report the hub's own
    # gene as a flank
    locus = _hub_locus('collapse', [Edge("hub", Strand.FORWARD, "hub", Strand.REVERSE)], {"hub": (100, 400)})

    assert _flank_relations(locus.downstream_flanks) == [
        ("gene_b1", 200, 0), ("gene_b2", 200, 1), ("gene_b3", 250, 1)]


def test_a_link_out_of_the_middle_of_a_hub_is_not_a_branch_of_its_flank_search():
    # IS1 runs off a1, across the two-contig hub h1 -> h2, and into b1. 'h2+ -> b2+' leaves the hub's edge, but
    # 'h1+ -> stub+' leaves its middle, into a stub of one copy's variant of the IS
    lengths = {"a1": 5000, "h1": 500, "h2": 500, "b1": 1000, "b2": 1000, "stub": 1000}
    edges = [Edge("a1", Strand.FORWARD, "h1", Strand.FORWARD), Edge("h1", Strand.FORWARD, "h2", Strand.FORWARD),
             Edge("h2", Strand.FORWARD, "b1", Strand.FORWARD), Edge("h2", Strand.FORWARD, "b2", Strand.FORWARD),
             Edge("h1", Strand.FORWARD, "stub", Strand.FORWARD)]
    builder = _flank_builder(lengths, edges, {"b1": (500, 800), "b2": (300, 600), "stub": (10, 100)},
                             mode='collapse', depths={c: 30.0 if c in ("h1", "h2") else 10.0 for c in lengths})
    alignments = {
        "a1": _batch([("IS1", 1600, 0, 300, "a1", 5000, 4700, 5000, 300, 300)]),
        "b1": _batch([("IS1", 1600, 1300, 1600, "b1", 1000, 0, 300, 300, 300)]),
    }

    [locus] = builder.assemble(alignments, min_coverage=0.30, min_identity=0.0)

    assert [s.contig for s in locus.targets[0].segments] == ["a1", "h1", "h2", "b1"]
    assert _flank_relations(locus.downstream_flanks) == [("gene_b1", 200, 0), ("gene_b2", 300, 1)]


def test_a_branch_does_not_loop_back_into_the_stitched_path():
    # 'b2+ -> hub-' turns back into the hub on its other strand: walking it would report the hub's own gene as a flank
    locus = _hub_locus('collapse', [Edge("b2", Strand.FORWARD, "hub", Strand.REVERSE)], {"hub": (100, 400)})

    assert _flank_relations(locus.downstream_flanks) == [
        ("gene_b1", 200, 0), ("gene_b2", 200, 1), ("gene_b3", 250, 1)]


def test_variant_mode_walks_flanks_only_from_the_ends_of_the_stitched_path():
    locus = _hub_locus('variant')

    assert _flank_relations(locus.downstream_flanks) == [("gene_b1", 200, 0)]
    assert _flank_relations(locus.upstream_flanks) == [("gene_a1", 200, 0)]


@pytest.mark.parametrize("depths, expected", [(None, 1.0), ({}, None)])
def test_fractional_depth_is_none_when_the_assembly_has_no_depth(depths, expected):
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0)]
    builder = _builder({"ctgA": 1000, "ctgB": 1000, "ctgC": 5000}, edges, depths=depths)
    alignments = {
        "ctgA": _batch([("IS1", 1000, 0, 500, "ctgA", 1000, 500, 1000, 500, 500)]),
        "ctgB": _batch([("IS1", 1000, 500, 1000, "ctgB", 1000, 0, 500, 500, 500)]),
        "ctgC": _batch([("IS2", 1000, 0, 1000, "ctgC", 5000, 1000, 2000, 1000, 1000)]),
    }

    loci = list(builder.assemble(alignments, min_coverage=0.30))

    assert len(loci) == 2 and {locus.fractional_depth for locus in loci} == {expected}


# Flank search characterization (#77) -------------------------------------------------------------------------------
# Today's behaviour, quirks included, pinned before the flank search moves into its own module. #78 fixes the quirks.
def _genes_builder(lengths, edges, gene_spans, hops, mode='variant', **builder_args):
    """gene_spans: contig -> [(start, end)] of its genes, read on '+', in start order; named gene_<contig>_<start>."""
    intervals = {c: IntervalBatch(np.array([s for s, _ in spans]), np.array([e for _, e in spans]),
                                  np.array([Strand.FORWARD] * len(spans), dtype=np.int8))
                 for c, spans in gene_spans.items()}
    features = {c: [GenomicFeature(f"gene_{c}_{s}", FeatureType.CDS, [LocationSegment(c, s, e, Strand.FORWARD)])
                    for s, e in spans] for c, spans in gene_spans.items()}
    no_indels = {c: [SimpleNamespace(insertions=[], deletions=[]) for _ in spans] for c, spans in gene_spans.items()}
    return _builder(lengths, edges, mode=mode, intervals=intervals, max_feature_hops=hops, features=features,
                    genes=no_indels, **builder_args)


def test_hop_0_flanks_come_nearest_first():
    builder = _genes_builder({"ctgA": 5000}, [], {"ctgA": [(100, 200), (300, 400), (1500, 1600), (1700, 1800)]}, 2)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 5000, 1000, 1200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == [("gene_ctgA_300", 600), ("gene_ctgA_100", 800)]
    assert _flank_genes(locus.downstream_flanks) == [("gene_ctgA_1500", 300), ("gene_ctgA_1700", 500)]


def test_a_hubs_flanks_come_hop_0_first_then_in_walk_order():
    locus = _hub_locus('collapse')

    assert [(f.feature.id, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_b1", 0), ("gene_b2", 1), ("gene_b3", 1)]


def test_quirk_a_claimed_gene_still_takes_a_slot_of_the_flank_budget():
    # ctgB's start links to both ends of ctgA, so both sides' walks enter ctgB on '+'. Upstream, walked first, claims
    # gene_ctgB_100; downstream cuts ctgB's genes to its budget of 1 before skipping the claimed one, so never reaches
    # gene_ctgB_300
    builder = _genes_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD),
                              Edge("ctgB", Strand.REVERSE, "ctgA", Strand.FORWARD)],
                             {"ctgB": [(100, 200), (300, 400)]}, 1)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 400, 600, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == [("gene_ctgB_100", 500)]
    assert locus.downstream_flanks == []


def test_quirk_a_gene_reached_back_across_the_origin_is_claimed_even_when_dropped():
    # The IS is on '-', so upstream, walked first, reads chrom forwards: it finds both genes at hop 0, then wraps round
    # and charges gene_1000 to its last budget slot before dropping it, as it lies past the target. Downstream,
    # wrapping the other way, then skips gene_1000 as claimed
    builder = _genes_builder({"chrom": 10000}, [Edge("chrom", Strand.FORWARD, "chrom", Strand.FORWARD)],
                             {"chrom": [(1000, 1500), (3000, 3500)]}, 3, circular={"chrom"},
                             max_flank_distance=None)
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 10000, 500, 700, 200, 200)], [Strand.REVERSE])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == [("gene_chrom_1000", 300), ("gene_chrom_3000", 2300)]
    assert [(f.feature.id, f.distance_bp, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_chrom_3000", 7000, 1)]


def test_quirk_upstream_hop_0_misses_a_gene_when_overlapping_genes_leave_ends_unsorted():
    # Sorted by start, the gene ends run 300, 1100, 400: the binary search on them for the target start lands
    # between the first two, so gene_ctgA_350, nearest, is never looked at
    builder = _genes_builder({"ctgA": 5000}, [], {"ctgA": [(100, 300), (200, 1100), (350, 400)]}, 3)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 5000, 1000, 1200, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert _flank_genes(locus.upstream_flanks) == [("gene_ctgA_100", 700)]
    assert _flank_genes(locus.passengers) == [("gene_ctgA_200", 0)]


def test_quirk_a_hop_0_gene_is_never_claimed_so_flanks_both_sides_of_a_small_circle():
    builder = _genes_builder({"chrom": 3000}, [Edge("chrom", Strand.FORWARD, "chrom", Strand.FORWARD)],
                             {"chrom": [(2000, 2500)]}, 1, circular={"chrom"}, max_flank_distance=None)
    alignments = {"chrom": _batch([("IS1", 200, 0, 200, "chrom", 3000, 500, 700, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [(f.feature.id, f.distance_bp, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_chrom_2000", 1300, 0)]
    assert [(f.feature.id, f.distance_bp, f.topological_dist) for f in locus.upstream_flanks] == [
        ("gene_chrom_2000", 1000, 1)]


def test_quirk_a_link_past_the_max_flank_distance_is_not_marked_visited():
    # ctgX is reached first through the long ctgP, past the limit, then through the short ctgQ, within it
    builder = _genes_builder({"ctgA": 1000, "ctgP": 5000, "ctgQ": 100, "ctgX": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgP", Strand.FORWARD),
                              Edge("ctgA", Strand.FORWARD, "ctgQ", Strand.FORWARD),
                              Edge("ctgP", Strand.FORWARD, "ctgX", Strand.FORWARD),
                              Edge("ctgQ", Strand.FORWARD, "ctgX", Strand.FORWARD)],
                             {"ctgX": [(100, 400)]}, 1, max_flank_distance=1000)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [(f.feature.id, f.distance_bp, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_ctgX_100", 200, 2)]


def test_quirk_a_hop_0_gap_is_measured_from_the_alignment_not_where_the_target_ends():
    # The target runs 200 bp past its alignment's end at 995, onto ctgB, yet the gene at 997 is 2 bp away
    builder = _genes_builder({"ctgA": 1000, "ctgB": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=100)],
                             {"ctgA": [(997, 1000)], "ctgB": [(500, 800)]}, 1)
    alignments = {"ctgA": _batch([("IS1", 500, 0, 300, "ctgA", 1000, 695, 995, 300, 300)])}

    [locus] = builder.assemble(alignments)

    assert _relation_summary(locus.downstream_flanks) == [("gene_ctgA_997", "DOWNSTREAM", 2, "NONE")]


def test_variant_mode_siblings_draw_down_one_budget_and_children_inherit_what_remains():
    # ctgP, visited first, takes one slot, so ctgQ gets only the other of its two genes; ctgR, past ctgP, still gets
    # the slot ctgP left when it was visited, as each contig queued carries the budget left at the time
    builder = _genes_builder({"ctgA": 1000, "ctgP": 1000, "ctgQ": 1000, "ctgR": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgP", Strand.FORWARD),
                              Edge("ctgA", Strand.FORWARD, "ctgQ", Strand.FORWARD),
                              Edge("ctgP", Strand.FORWARD, "ctgR", Strand.FORWARD)],
                             {"ctgP": [(100, 200)], "ctgQ": [(100, 200), (300, 400)], "ctgR": [(100, 200)]}, 2)
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [(f.feature.id, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_ctgP_100", 1), ("gene_ctgQ_100", 1), ("gene_ctgR_100", 2)]


def test_collapse_mode_resets_the_budget_for_every_link():
    builder = _genes_builder({"ctgA": 1000, "ctgP": 1000, "ctgQ": 1000, "ctgR": 1000},
                             [Edge("ctgA", Strand.FORWARD, "ctgP", Strand.FORWARD),
                              Edge("ctgA", Strand.FORWARD, "ctgQ", Strand.FORWARD),
                              Edge("ctgP", Strand.FORWARD, "ctgR", Strand.FORWARD)],
                             {"ctgP": [(100, 200)], "ctgQ": [(100, 200), (300, 400)], "ctgR": [(100, 200)]}, 2,
                             mode='collapse')
    alignments = {"ctgA": _batch([("IS1", 200, 0, 200, "ctgA", 1000, 800, 1000, 200, 200)])}

    [locus] = builder.assemble(alignments)

    assert [(f.feature.id, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_ctgP_100", 1), ("gene_ctgQ_100", 1), ("gene_ctgQ_300", 1), ("gene_ctgR_100", 2)]


def test_a_stitched_paths_walk_may_reenter_the_path_when_its_hub_has_no_branches():
    # Nothing else leaves the hub, so no contig is blocked: the walk off b1 loops back to a1's start
    lengths = {"a1": 5000, "hub": 1000, "b1": 1000}
    edges = [Edge("a1", Strand.FORWARD, "hub", Strand.FORWARD), Edge("hub", Strand.FORWARD, "b1", Strand.FORWARD),
             Edge("b1", Strand.FORWARD, "a1", Strand.FORWARD)]
    builder = _genes_builder(lengths, edges, {"a1": [(4000, 4500)]}, 1, mode='collapse',
                             depths={"hub": 30.0, "a1": 10.0, "b1": 10.0}, max_flank_distance=None)
    alignments = {
        "a1": _batch([("IS1", 1600, 0, 300, "a1", 5000, 4700, 5000, 300, 300)]),
        "b1": _batch([("IS1", 1600, 1300, 1600, "b1", 1000, 0, 300, 300, 300)]),
    }

    [locus] = builder.assemble(alignments, min_coverage=0.30, min_identity=0.0)

    assert [s.contig for s in locus.targets[0].segments] == ["a1", "hub", "b1"]
    assert [(f.feature.id, f.distance_bp, f.topological_dist) for f in locus.downstream_flanks] == [
        ("gene_a1_4000", 4700, 1)]
