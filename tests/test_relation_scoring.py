import random
from types import SimpleNamespace

from Bio.Seq import reverse_complement

from eris.constants import Context, Effect
from isoforms import read_isoform_groups
from relation_scoring import (GenePlacer, ISCopy, Placement, TruthRelation, expected_relations, read_is_copies,
                              read_truth, score_assignment, score_relations)

GROUPS = {"ISA": "grpA", "ISB": "grpB"}


def relation(gene_id, spatial, dist=0, topo=0, effect=Effect.NONE, contig="ctg"):
    feature = SimpleNamespace(id=gene_id, segments=[SimpleNamespace(contig=contig)])
    return SimpleNamespace(feature=feature, spatial=spatial, distance_bp=dist, topological_dist=topo, effect=effect)


def locus(name, passengers=(), up=(), down=(), id_=None):
    return SimpleNamespace(id=id_ or f"locus_{name}", targets=[SimpleNamespace(id=f"{name}_fam_grp")],
                           passengers=list(passengers), upstream_flanks=list(up), downstream_flanks=list(down))


def truth(side, tag, start, end, group="grpA", strand="+", distance=10, effect="none", partial=False,
          scored=True):
    return TruthRelation(group=group, side=side, locus_tag=tag, replicon="chr", start=start, end=end,
                         strand=strand, partial=partial, pseudo=False, distance_bp=distance,
                         effect=effect if scored else None)


def placed(**genes):
    """gene id -> (start, end[, strand[, identity]]) on 'chr'; genes left out are unplaceable."""
    table = {g: [Placement("chr", *(v + ("+", 1.0)[len(v) - 2:]))] for g, v in genes.items()}
    return lambda gene_id: table.get(gene_id, [])


def test_relation_is_found_by_group_side_and_overlapping_gene():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000)]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM, 30)], down=[relation("g2", Context.DOWNSTREAM, 40)])]
    # g1 covers half of T1; g2 lies on the other strand of T2
    score = score_relations(truths, loci, GROUPS, placed(g1=(1500, 2600), g2=(5000, 6000, "-")))

    assert score.found == {("grpA", "upstream", "T1")}
    assert score.recall == {"upstream": 1.0, "downstream": 0.0}


def test_gene_crossing_an_is_end_is_a_flank_on_the_side_it_sticks_out_of():
    truths = [truth("upstream", "T1", 1000, 2000, distance=-20), truth("downstream", "T2", 5000, 6000, distance=-5)]
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START), relation("g2", Context.OVERLAPPING_END),
                                     relation("tnp", Context.INSIDE)])]
    score = score_relations(truths, loci, GROUPS, placed(g1=(1000, 2000), g2=(5000, 6000), tnp=(2000, 5000)))

    assert score.recall == {"upstream": 1.0, "downstream": 1.0}


def test_pooled_recall_forgives_a_flank_found_on_the_wrong_side():
    # The gene was found, but eris read the IS on the other strand
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000)]
    loci = [locus("ISA", down=[relation("g1", Context.DOWNSTREAM)])]
    score = score_relations(truths, loci, GROUPS, placed(g1=(1000, 2000)))

    assert score.recall == {"upstream": 0.0, "downstream": 0.0}
    assert score.pooled_recall == 0.5


def test_flank_of_another_group_does_not_count():
    truths = [truth("upstream", "T1", 1000, 2000)]
    loci = [locus("ISB", up=[relation("g1", Context.UPSTREAM)])]
    assert score_relations(truths, loci, GROUPS, placed(g1=(1000, 2000))).found == set()


def test_flank_of_another_isoform_group_does_not_count_even_in_the_same_family(tmp_path):
    # ISSen10 and IS629 are both IS3: per family, an IS629 locus's flank was credited to ISSen10 (#42)
    row = {"is_id": "ISSen10_1", "replicon": "chr", "side": "upstream", "locus_tag": "T1", "cds_start": "1000",
           "cds_end": "2000", "cds_strand": "+", "partial": "False", "pseudo": "False", "distance_bp": "90",
           "effect": "none", "inside_is": ""}
    (tmp_path / "truth.tsv").write_text("\t".join(row) + "\n" + "\t".join(row.values()) + "\n")
    loci = [locus("IS629", up=[relation("g1", Context.UPSTREAM)])]
    place = placed(g1=(1000, 2000))

    by_family = {"ISSen10": "IS3", "IS629": "IS3"}
    assert score_relations(read_truth(tmp_path / "truth.tsv", by_family), loci, by_family, place).found == {
        ("IS3", "upstream", "T1")}
    groups = read_isoform_groups()
    assert score_relations(read_truth(tmp_path / "truth.tsv", groups), loci, groups, place).found == set()


def test_precision_scores_the_first_flank_of_each_branch():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T3", 7000, 8000)]
    local = locus("ISA", up=[relation("g1", Context.UPSTREAM, 10), relation("g2", Context.UPSTREAM, 900)])
    # A collapsed repeat: two flanking contigs one hop away, and a third further along one of them
    repeat = locus("ISA", down=[relation("g3", Context.DOWNSTREAM, 50, topo=1, contig="c1"),
                                relation("g4", Context.DOWNSTREAM, 800, topo=1, contig="c1"),
                                relation("g5", Context.DOWNSTREAM, 20, topo=1, contig="c2"),
                                relation("g6", Context.DOWNSTREAM, 3000, topo=2, contig="c3")])
    place = placed(g1=(1000, 2000), g2=(3000, 3500), g3=(7000, 8000), g4=(9000, 9500), g5=(20000, 21000),
                   g6=(30000, 31000))
    score = score_relations(truths, [local, repeat], GROUPS, place)

    # First flanks: g1 and g3 are true, g5 is not; g2, g4 and g6 lie further along their branches
    assert score.precision == 2 / 3


def test_distance_band_agreement_on_found_relations():
    truths = [truth("upstream", "T1", 1000, 2000, distance=-12),  # overlapping, eris agrees
              truth("downstream", "T2", 5000, 6000, distance=140),  # near; eris says far
              truth("upstream", "T3", 9000, 9500, group="grpB", distance=400)]  # far, eris agrees
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START)],
                  down=[relation("g2", Context.DOWNSTREAM, 160)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM, 380)])]
    score = score_relations(truths, loci, GROUPS, placed(g1=(1000, 2000), g2=(5000, 6000), g3=(9000, 9500)))

    assert score.band_agreement == 2 / 3
    # Errors |eris - truth|, with overlaps read as 0 as eris reports them: 0, 20, 20
    assert (score.median_distance_error, score.max_distance_error) == (20, 20)


def test_disrupted_or_truncated_scores_against_truth_disrupted():
    truths = [truth("upstream", "T1", 1000, 2000, effect="disrupted"),
              truth("downstream", "T2", 5000, 6000, effect="disrupted"),
              truth("upstream", "T3", 9000, 9500, group="grpB", effect="disrupted"),
              truth("downstream", "T4", 12000, 13000, group="grpB", scored=False),  # inside another IS
              truth("upstream", "T5", 20000, 21000)]
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START, effect=Effect.DISRUPTED)],
                  down=[relation("g2", Context.DOWNSTREAM, 3, effect=Effect.TRUNCATED)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM, 10)],
                  passengers=[relation("g4", Context.OVERLAPPING_END, effect=Effect.DISRUPTED)]),
            locus("ISA", up=[relation("g5", Context.UPSTREAM, 0, effect=Effect.DISRUPTED)],
                  down=[relation("g6", Context.DOWNSTREAM, 40, effect=Effect.UPREGULATED)])]
    place = placed(g1=(1000, 2000), g2=(5000, 6000), g3=(9000, 9500), g4=(12000, 13000), g5=(20000, 21000),
                   g6=(30000, 31000))
    score = score_relations(truths, loci, GROUPS, place)

    # g1 and g2 are right, g5 is a false call, g3 is missed; g4's effect is not scored
    assert (score.disrupted_precision, score.disrupted_recall) == (2 / 3, 2 / 3)
    assert score.upregulated == [("grpA", "downstream", "g6")]


def test_misses_found_with_a_larger_hops_budget_are_told_apart():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000),
              truth("downstream", "T3", 9000, 9500)]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM)], down=[relation("short", Context.DOWNSTREAM)])]
    # With a larger budget the same locus also reaches g2, the true downstream flank; T3 stays out of reach
    reach = [locus("ISA", up=[relation("g1", Context.UPSTREAM)],
                   down=[relation("short", Context.DOWNSTREAM), relation("g2", Context.DOWNSTREAM, 900)])]
    score = score_relations(truths, loci, GROUPS, placed(g1=(1000, 2000), g2=(5000, 6000)), reach_loci=reach)

    assert score.miss_causes == {("grpA", "downstream", "T2"): "budget", ("grpA", "downstream", "T3"): "gene"}
    assert score.unplaceable == ["short"]


def test_nucleotide_check_needs_identity_and_coverage_of_complete_cds():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000),
              truth("upstream", "T3", 9000, 10000, group="grpB", partial=True),
              truth("downstream", "T4", 12000, 13000, group="grpB")]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM)], down=[relation("g2", Context.DOWNSTREAM)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM)], down=[relation("g4", Context.DOWNSTREAM)])]
    place = placed(g1=(1000, 2000, "+", 0.98),  # too many mismatches
                   g2=(5300, 6000),  # covers 70% of the CDS
                   g3=(9400, 10000),  # 60%, but the RefSeq CDS is partial
                   g4=(12000, 13000, "+", 0.995))
    score = score_relations(truths, loci, GROUPS, place)

    assert score.found == {t.key for t in truths}
    assert score.nucleotide_mismatches == {("grpA", "upstream", "T1"), ("grpA", "downstream", "T2")}


def test_genes_are_placed_by_alignment_or_exact_search(tmp_path):
    rng = random.Random(15)
    chrom = "".join(rng.choice("ACGT") for _ in range(20000))
    (tmp_path / "ref.fasta").write_text(f">chr\n{chrom}\n")
    placer = GenePlacer(tmp_path / "ref.fasta")

    assert placer.place(chrom[5000:5600].encode()) == [Placement("chr", 5000, 5600, "+", 1.0)]
    assert placer.place(reverse_complement(chrom[10000:10900]).encode()) == [Placement("chr", 10000, 10900, "-", 1.0)]
    # Too short for minimap2
    assert placer.place(reverse_complement(chrom[15000:15060]).encode()) == [Placement("chr", 15000, 15060, "-", 1.0)]
    assert placer.place(b"ACGT" * 15) == []


# Assignment score (#52) ---------------------------------------------------------------------------------------------
def copy(name, start, end, strand="+", group=None):
    return ISCopy(name=name, group=group or GROUPS[name.split("_")[0]], replicon="chr", start=start, end=end,
                  strand=strand)


def at(c: ISCopy) -> Placement:
    return Placement(c.replicon, c.start, c.end, c.strand, 1.0)


def covering(**by_locus):
    """locus id -> the truth copies its IS segments place onto."""
    return lambda lc: [at(c) for c in by_locus.get(lc.id, [])]


def errors(score, kind):
    return {e for e in score.errors if e[1].startswith(kind)}


A1, B1 = copy("ISA_1", 10000, 11000), copy("ISB_1", 50000, 51000, "-")


def test_expected_relations_are_the_nearest_eris_gene_on_each_side_of_each_copy():
    place = placed(a_far=(7000, 8500), a_up=(9000, 9900), a_up_dup=(9000, 9890),  # the same gene, called twice
                   tnp=(10100, 10900),  # inside the IS: not a flank
                   a_down=(10980, 12000),  # crosses the IS's end
                   b_left=(48000, 49990), b_right=(51050, 52000))  # B1 is read on -, so its upstream is to the right
    genes = ["a_far", "a_up", "a_up_dup", "tnp", "a_down", "b_left", "b_right", "unplaced"]
    expected = {(e.key, e.genes, e.distances) for e in expected_relations([A1, B1], genes, place)}

    assert expected == {(("grpA", "upstream", "a_up"), frozenset({"a_up", "a_up_dup"}), frozenset({100})),
                        (("grpA", "downstream", "a_down"), frozenset({"a_down"}), frozenset({-20})),
                        (("grpB", "upstream", "b_right"), frozenset({"b_right"}), frozenset({50})),
                        (("grpB", "downstream", "b_left"), frozenset({"b_left"}), frozenset({10}))}


def test_a_neighbouring_copys_transposase_is_the_nearest_gene_of_an_adjacent_pair():
    # ISA_1 and ISB_2 touch end to end; each one's transposase is the gene next to the other (ADR 0005)
    b2 = copy("ISB_2", 11000, 12000)
    place = placed(tnp_a=(10100, 10900), tnp_b=(11100, 11900), beyond=(12100, 13000))
    genes = ["tnp_a", "tnp_b", "beyond"]
    own = {e.key: e.distances for e in expected_relations([A1, b2], genes, place)}
    any_copy = {e.key: e.distances for e in expected_relations([A1, b2], genes, place, universe="any_copy")}

    assert own[("grpA", "downstream", "tnp_b")] == frozenset({100})
    assert own[("grpB", "upstream", "tnp_a")] == frozenset({100})
    assert ("grpA", "downstream", "beyond") in any_copy and not any(k[2].startswith("tnp") for k in any_copy)


def test_expected_relations_wrap_around_a_circular_replicon():
    near_origin = copy("ISA_1", 500, 1500)
    place = placed(g_end=(99000, 99800), g_mid=(60000, 61000))
    expected = expected_relations([near_origin], ["g_end", "g_mid"], place, lengths={"chr": 100000})

    assert {e.key: e.distances for e in expected} == {("grpA", "upstream", "g_end"): frozenset({700}),
                                                      ("grpA", "downstream", "g_mid"): frozenset({58500})}


def test_each_wrong_first_flank_gets_one_cause_and_overcalls_leave_the_headline():
    place = placed(a_far=(7000, 8500), a_up=(9000, 9900), a_down=(11100, 12000), a_next=(12500, 13000),
                   b_left=(48000, 49990), b_right=(51050, 52000))
    genes = ["a_far", "a_up", "a_down", "a_next", "b_left", "b_right"]
    loci = [locus("ISA", up=[relation("a_up", Context.UPSTREAM)], down=[relation("b_left", Context.DOWNSTREAM)]),
            locus("ISA", down=[relation("a_next", Context.DOWNSTREAM)], id_="locus_ISA_2"),
            locus("ISB", up=[relation("b_left", Context.UPSTREAM)], down=[relation("short", Context.DOWNSTREAM)]),
            locus("ISC", up=[relation("a_far", Context.UPSTREAM)])]  # no truth copy of ISC
    cover = covering(locus_ISA=[A1], locus_ISA_2=[A1], locus_ISB=[B1])
    score = score_assignment([A1, B1], genes, loci, GROUPS, place, cover)

    assert errors(score, "FP") == {(("grpA", "downstream", "b_left"), "FP_group"),
                                   (("grpA", "downstream", "a_next"), "FP_not_nearest"),
                                   (("grpB", "upstream", "b_left"), "FP_side"),
                                   (("grpB", "downstream", "short"), "FP_unplaced"),
                                   (("ISC", "upstream", "a_far"), "FP_overcall")}
    assert (score.first_flanks, score.overcalls) == (6, 1)
    assert score.precision("nearest") == 1 / 5


def test_each_missed_relation_gets_one_cause_and_undetected_copies_leave_the_headline():
    a2, b2 = copy("ISA_2", 30000, 31000), copy("ISB_2", 70000, 71000, "-")
    place = placed(a_up=(9000, 9900), a_down=(11100, 12000), a2_up=(29000, 29950), a2_down=(31050, 32000),
                   b_left=(48000, 49990), b_right=(51050, 52000), b2_left=(69000, 69900))
    genes = ["a_up", "a_down", "a2_up", "a2_down", "b_left", "b_right", "b2_left"]
    loci = [locus("ISA", up=[relation("a_up", Context.UPSTREAM), relation("a_down", Context.UPSTREAM, 1100)],
                  down=[relation("b_left", Context.DOWNSTREAM)]),
            locus("ISB")]
    reach = [locus("ISB", up=[relation("b_right", Context.UPSTREAM, topo=5)])]
    cover = covering(locus_ISA=[A1], locus_ISB=[B1, b2])  # nothing covers ISA_2
    score = score_assignment([A1, a2, B1, b2], genes, loci, GROUPS, place, cover, reach_loci=reach)

    assert errors(score, "FN") == {(("grpA", "downstream", "a_down"), "FN_side"),
                                   (("grpA", "upstream", "a2_up"), "FN_undetected"),
                                   (("grpA", "downstream", "a2_down"), "FN_undetected"),
                                   (("grpB", "downstream", "b_left"), "FN_group"),
                                   (("grpB", "upstream", "b_right"), "FN_budget"),
                                   (("grpB", "downstream", "b2_left"), "FN_unreported")}
    assert (score.expected, score.undetected) == (7, 2)
    assert score.recall("nearest") == 1 / 5


def test_a_repeat_hub_expects_the_nearest_genes_of_all_its_copies():
    a2 = copy("ISA_2", 30000, 31000)
    place = placed(a_up=(9000, 9900), a_down=(11100, 12000), a2_up=(29000, 29950), a2_down=(31050, 32000))
    # One repeat contig, a flanking contig per copy on each side
    hub = locus("ISA", up=[relation("a_up", Context.UPSTREAM, topo=1, contig="c1"),
                           relation("a2_up", Context.UPSTREAM, topo=1, contig="c2")],
                down=[relation("a_down", Context.DOWNSTREAM, topo=1, contig="c3"),
                      relation("a2_down", Context.DOWNSTREAM, topo=1, contig="c4")])
    score = score_assignment([A1, a2], ["a_up", "a_down", "a2_up", "a2_down"], [hub], GROUPS, place,
                             covering(locus_ISA=[A1, a2]))

    assert (score.precision("nearest"), score.recall("nearest")) == (1.0, 1.0)
    assert score.errors == set()


def test_tiers_add_nearest_then_distance():
    place = placed(a_far=(7000, 8500), a_up=(9000, 9900), a_down=(11000, 12000))
    # a_far is on the right side but not the nearest; a_down is the nearest, but eris puts it far rather than near
    loci = [locus("ISA", up=[relation("a_far", Context.UPSTREAM, 1500)],
                  down=[relation("a_down", Context.DOWNSTREAM, 400)])]
    score = score_assignment([A1], ["a_far", "a_up", "a_down"], loci, GROUPS, place, covering(locus_ISA=[A1]))

    assert [score.precision(t) for t in ("side", "nearest", "distance")] == [1.0, 0.5, 0.0]
    assert [score.recall(t) for t in ("side", "nearest", "distance")] == [0.5, 0.5, 0.0]
    assert (score.band_agreement, score.distance_errors) == (0.0, [400])
    assert [score.f1(t) for t in ("side", "nearest", "distance")] == [2 / 3, 0.5, 0.0]


def test_side_tier_credits_every_flank_the_nearest_tier_credits():
    # Between two tandem copies, g is the nearest gene upstream of ISA_2, though nearer to ISA_1's downstream end
    a2 = copy("ISA_2", 12000, 13000)
    loci = [locus("ISA", up=[relation("g", Context.UPSTREAM, 500)])]
    score = score_assignment([A1, a2], ["g"], loci, GROUPS, placed(g=(11100, 11500)), covering(locus_ISA=[a2]))

    assert (score.precision("side"), score.precision("nearest")) == (1.0, 1.0)


def test_distance_of_a_hub_flank_is_checked_against_the_copy_it_flanks():
    # A gene repeated next to two copies: 50 bp upstream of one, 400 bp upstream of the other
    a2 = copy("ISA_2", 30000, 31000)
    copies = {"rep": [Placement("chr", 9000, 9950, "+", 1.0), Placement("chr", 28700, 29600, "+", 1.0)]}
    hub = locus("ISA", up=[relation("rep", Context.UPSTREAM, 400, topo=1)])
    score = score_assignment([A1, a2], ["rep"], [hub], GROUPS, lambda g: copies.get(g, []),
                             covering(locus_ISA=[A1, a2]))

    assert (score.precision("distance"), score.distance_errors) == (1.0, [0])


def test_is_copies_are_read_from_the_bed(tmp_path):
    (tmp_path / "is.bed").write_text("chr\t100\t900\tISA_1\t0\t+\nchr\t2000\t2800\tISB\t0\t-\npl\t5\t50\tISA_2\t0\t+\n")

    assert read_is_copies(tmp_path / "is.bed", GROUPS, {"chr"}) == [
        ISCopy("ISA_1", "grpA", "chr", 100, 900, "+"), ISCopy("ISB", "grpB", "chr", 2000, 2800, "-")]


def test_part_of_a_sequence_is_placed_through_its_context(tmp_path):
    # An IS segment is placed with the contig sequence around it, so identical copies are told apart
    rng = random.Random(52)
    chrom = "".join(rng.choice("ACGT") for _ in range(20000))
    (tmp_path / "ref.fasta").write_text(f">chr\n{chrom}\n")
    placer = GenePlacer(tmp_path / "ref.fasta")

    assert placer.place_part(chrom[4000:7000].encode(), 1000, 2000) == [Placement("chr", 5000, 6000, "+", 1.0)]
    assert placer.place_part(reverse_complement(chrom[4000:7000]).encode(), 1000, 2000) == [
        Placement("chr", 5000, 6000, "-", 1.0)]
    assert placer.lengths == {"chr": 20000}
