"""
Scores eris's relations against the EC958 relation truth (built by scripts/build_relation_truth.py).

eris genes are pyfgs predictions on the assembly, so each one is first placed on the reference genome; a gene then
matches a RefSeq CDS when it lies on the same replicon and strand and overlaps it by at least half of the shorter
of the two. A truth relation is keyed by (group, side, locus tag) and is found when any eris locus of that group
reports a matching flank on that side. The benchmark groups IS names by isoform group (docs/adr/0004), and prints
scores per IS family for reference. See docs/adr/0002-relation-benchmark-truth.md.

`score_assignment` is a second, geometric score (#52) that leaves gene calling out: the expected flank on each side of
each truth IS copy is the nearest of eris's own genes on the reference, and an eris gene is matched by its id.
"""
import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import mappy
from Bio import SeqIO
from Bio.Seq import reverse_complement

from eris.constants import Context, Effect
from eris.pipeline import FeatureRelation

SIDES = ('upstream', 'downstream')
# A gene crossing an end of the IS is reported as a passenger, but it is a flank on the side it sticks out of
OVERLAP_SIDES = {Context.OVERLAPPING_START: 'upstream', Context.OVERLAPPING_END: 'downstream'}
MIN_OVERLAP = 0.5  # of the shorter of the eris gene and the RefSeq CDS
DISRUPTING = Effect.DISRUPTED | Effect.TRUNCATED  # a truncation is a kind of disruption
MIN_IDENTITY = 0.99  # over the aligned part, for a matched flank's sequence to count as correct
MIN_CDS_COVERAGE = 0.8  # of the RefSeq CDS; waived when it is partial or pseudo
RelationKey = tuple[str, str, str]  # (group, side, locus tag)
NEAR = 150  # bp; pyfgs start codons and eris's IS ends differ from RefSeq by a few to tens of bp, so distances are banded


@dataclass(frozen=True)
class TruthRelation:
    group: str
    side: str
    locus_tag: str
    replicon: str
    start: int  # 0-based, half-open
    end: int
    strand: str
    partial: bool
    pseudo: bool
    distance_bp: int  # negative when the CDS overlaps the IS
    effect: str | None  # 'disrupted' or 'none'; None when the effect is not scored

    @property
    def key(self) -> RelationKey:
        return self.group, self.side, self.locus_tag


@dataclass(frozen=True)
class Placement:
    """Where an eris gene lies on the reference: `strand` is the strand its coding sequence reads on."""
    replicon: str
    start: int
    end: int
    strand: str
    identity: float  # over the aligned part


def read_truth(path: Path, groups: dict[str, str], replicons: set[str] | None = None) -> list[TruthRelation]:
    """
    Reads the flank relations of the truth TSV, optionally only those on the given replicons. The effect of a flank
    that lies inside another IS, or whose curated effect is uncertain, is not scored.
    """
    with open(path) as handle:
        rows = [r for r in csv.DictReader(handle, delimiter='\t') if r['side'] in SIDES]
    truths = []
    for r in rows:
        if replicons is not None and r['replicon'] not in replicons:
            continue
        scored = not r['inside_is'] and r['effect'] in ('disrupted', 'none')
        truths.append(TruthRelation(
            group=group_of(r['is_id'], groups), side=r['side'], locus_tag=r['locus_tag'], replicon=r['replicon'],
            start=int(r['cds_start']), end=int(r['cds_end']), strand=r['cds_strand'], partial=r['partial'] == 'True',
            pseudo=r['pseudo'] == 'True', distance_bp=int(r['distance_bp']), effect=r['effect'] if scored else None))
    return truths


class GenePlacer:
    """
    Places eris genes on the reference genome by aligning their coding sequence with minimap2. A gene that aligns
    equally well to several places (e.g. a repeated gene) gets all of them; one too short to align is looked up by
    exact search.
    """

    def __init__(self, reference: Path):
        self._aligner = mappy.Aligner(str(reference), preset='asm5')
        self._replicons = {r.id: str(r.seq).upper() for r in SeqIO.parse(reference, 'fasta')}

    @property
    def lengths(self) -> dict[str, int]:
        return {name: len(seq) for name, seq in self._replicons.items()}

    def place(self, sequence: bytes) -> list[Placement]:
        seq = sequence.decode().upper()
        hits = self._best_hits(seq)
        if hits:
            return [Placement(h.ctg, h.r_st, h.r_en, '+' if h.strand == 1 else '-', h.mlen / h.blen) for h in hits]
        return self._search(seq)

    def place_part(self, sequence: bytes, start: int, end: int) -> list[Placement]:
        """
        Places `sequence[start:end]` by aligning all of `sequence`, so the sequence around the part can tell apart
        identical copies of it. Reference coordinates are read off the alignment's ends, ignoring indels.
        """
        seq = sequence.decode().upper()
        placements = []
        for h in self._best_hits(seq):
            a, b = max(start, h.q_st) - h.q_st, min(end, h.q_en) - h.q_st
            if a >= b:
                continue
            if h.strand == 1:
                ref_start, ref_end = h.r_st + a, h.r_st + b
            else:
                ref_start, ref_end = h.r_en - b, h.r_en - a
            placements.append(Placement(h.ctg, max(h.r_st, ref_start), min(h.r_en, ref_end),
                                        '+' if h.strand == 1 else '-', h.mlen / h.blen))
        return placements or self._search(seq[start:end])

    def _best_hits(self, seq: str) -> list:
        hits = list(self._aligner.map(seq))
        best = max((h.mlen for h in hits), default=0)
        return [h for h in hits if h.mlen == best]

    def _search(self, seq: str) -> list[Placement]:
        placements = []
        for name, replicon in self._replicons.items():
            for strand, query in (('+', seq), ('-', reverse_complement(seq))):
                start = replicon.find(query)
                while start != -1:
                    placements.append(Placement(name, start, start + len(seq), strand, 1.0))
                    start = replicon.find(query, start + 1)
        return placements


@dataclass(frozen=True)
class Flank:
    """One flank reported by eris, read from a locus."""
    group: str
    side: str
    relation: FeatureRelation


@dataclass
class RelationScore:
    total: dict[str, int] = field(default_factory=dict)  # truth relations per side
    found: set[RelationKey] = field(default_factory=set)
    misses: set[RelationKey] = field(default_factory=set)
    budget_misses: set[RelationKey] = field(default_factory=set)  # misses found with a larger hops budget
    nucleotide_mismatches: set[RelationKey] = field(default_factory=set)  # found, but the sequence is off
    found_any_side: set[RelationKey] = field(default_factory=set)  # the gene was found, maybe on the other side
    first_flanks: int = 0  # eris's first flank per side per branch
    true_first_flanks: int = 0  # ... that match a truth relation
    bands: list[tuple[str, str]] = field(default_factory=list)  # (truth, eris) distance band of each found relation
    distance_errors: list[int] = field(default_factory=list)
    # Disrupted precision is over eris's first flanks called DISRUPTED or TRUNCATED, like `precision`; a disrupted
    # call further along a branch is not counted. Recall is over the truth's disrupted relations.
    disrupted_called: int = 0
    disrupted_called_true: int = 0
    disrupted_truth: int = 0
    disrupted_truth_found: int = 0
    upregulated: list[tuple[str, str, str]] = field(default_factory=list)  # (group, side, eris gene); not scored
    unplaceable: list[str] = field(default_factory=list)  # reported flank genes with no place on the reference

    @property
    def miss_causes(self) -> dict[RelationKey, str]:
        """Each miss with its cause: 'budget' if a larger hops budget finds it, else 'gene'."""
        return {k: 'budget' if k in self.budget_misses else 'gene' for k in self.misses}

    @property
    def recall(self) -> dict[str, float]:
        return {side: sum(k[1] == side for k in self.found) / n if n else 0.0 for side, n in self.total.items()}

    @property
    def pooled_recall(self) -> float:
        """Separates finding the gene from reading the IS strand right."""
        n = sum(self.total.values())
        return len(self.found_any_side) / n if n else 0.0

    @property
    def precision(self) -> float:
        return self.true_first_flanks / self.first_flanks if self.first_flanks else 0.0

    @property
    def disrupted_precision(self) -> float:
        return self.disrupted_called_true / self.disrupted_called if self.disrupted_called else 0.0

    @property
    def disrupted_recall(self) -> float:
        return self.disrupted_truth_found / self.disrupted_truth if self.disrupted_truth else 0.0

    @property
    def band_agreement(self) -> float:
        return sum(t == e for t, e in self.bands) / len(self.bands) if self.bands else 0.0

    @property
    def median_distance_error(self) -> float:
        return median(self.distance_errors) if self.distance_errors else 0.0

    @property
    def max_distance_error(self) -> int:
        return max(self.distance_errors, default=0)


def group_of(name: str, groups: dict[str, str]) -> str:
    """
    The group of a target or truth IS name in `groups`, which maps IS names to their isoform group or IS family,
    e.g. 'IS600_IS3_IS3' or 'IS600_2' -> 'IS600'. A name not in `groups` is its own group.
    """
    name = name.split('_')[0]
    return groups.get(name, name)


def locus_group(locus, groups: dict[str, str]) -> str:
    return group_of(locus.targets[0].id, groups)


def read_flanks(locus, groups: dict[str, str]) -> list[Flank]:
    """Every flank of a locus, nearest first on each side, including passengers that cross an end of the IS."""
    group = locus_group(locus, groups)
    flanks = [Flank(group, OVERLAP_SIDES[r.spatial], r) for r in locus.passengers if r.spatial in OVERLAP_SIDES]
    flanks += [Flank(group, 'upstream', r) for r in locus.upstream_flanks]
    flanks += [Flank(group, 'downstream', r) for r in locus.downstream_flanks]
    return flanks


def band(distance: int, overlapping: bool = False) -> str:
    if overlapping or distance < 0:
        return 'overlapping'
    return 'near' if distance <= NEAR else 'far'


def first_flanks(flanks: list[Flank]) -> list[Flank]:
    """
    The nearest flank of each branch on each side of one locus. A flank on the IS's own contig is the only branch on
    its side. Otherwise each contig at the fewest hops that carries a gene starts a branch; flanks further along are
    left out, since eris does not report which branch a deeper contig continues.
    """
    firsts = []
    for side in SIDES:
        on_side = [f for f in flanks if f.side == side]
        if not on_side:
            continue
        nearest_hop = min(f.relation.topological_dist for f in on_side)
        contigs = set()
        for f in on_side:
            contig = f.relation.feature.segments[0].contig
            if f.relation.topological_dist == nearest_hop and contig not in contigs:
                contigs.add(contig)
                firsts.append(f)
            if nearest_hop == 0:
                break
    return firsts


def overlap(placement: Placement, truth: TruthRelation) -> int:
    return min(placement.end, truth.end) - max(placement.start, truth.start)


def matches(placement: Placement, truth: TruthRelation) -> bool:
    if placement.replicon != truth.replicon or placement.strand != truth.strand:
        return False
    return overlap(placement, truth) >= MIN_OVERLAP * min(placement.end - placement.start, truth.end - truth.start)


def sequence_agrees(placement: Placement, truth: TruthRelation) -> bool:
    """Whether a matched eris gene has the RefSeq CDS's sequence; boundaries of partial or pseudo CDS are not checked."""
    covered = overlap(placement, truth) / (truth.end - truth.start)
    return placement.identity >= MIN_IDENTITY and (covered >= MIN_CDS_COVERAGE or truth.partial or truth.pseudo)


def matching_flanks(truth: TruthRelation, flanks: list[Flank], place: Callable[[str], list[Placement]]) -> list[Flank]:
    """Flanks of the truth relation's group, on either side, whose gene matches its CDS."""
    return [f for f in flanks if f.group == truth.group and any(matches(p, truth) for p in place(f.relation.feature.id))]


def score_relations(truths: list[TruthRelation], loci: list, groups: dict[str, str],
                    place: Callable[[str], list[Placement]], reach_loci: list | None = None) -> RelationScore:
    """
    Scores eris loci against the truth. `place` returns the reference placements of an eris gene by its id, or an
    empty list when the gene cannot be placed. `reach_loci` are the same assembly's loci from a run with a larger
    hops budget: a miss they find is put down to the budget rather than to a wrong gene.
    """
    per_locus = [read_flanks(locus, groups) for locus in loci]
    flanks = [f for locus_flanks in per_locus for f in locus_flanks]
    score = RelationScore(total={side: sum(t.side == side for t in truths) for side in SIDES})
    score.unplaceable = sorted({f.relation.feature.id for f in flanks if not place(f.relation.feature.id)})
    score.upregulated = [(f.group, f.side, f.relation.feature.id) for f in flanks
                         if f.relation.effect & Effect.UPREGULATED]
    for f in (f for locus_flanks in per_locus for f in first_flanks(locus_flanks)):
        placements = place(f.relation.feature.id)
        matched = [t for t in truths
                   if t.group == f.group and t.side == f.side and any(matches(p, t) for p in placements)]
        score.first_flanks += 1
        score.true_first_flanks += bool(matched)
        # A flank whose effect is not scored is left out
        if f.relation.effect & DISRUPTING and not any(t.effect is None for t in matched):
            score.disrupted_called += 1
            score.disrupted_called_true += any(t.effect == 'disrupted' for t in matched)
    for t in truths:
        matching = matching_flanks(t, flanks, place)
        if matching:
            score.found_any_side.add(t.key)
        on_side = [f.relation for f in matching if f.side == t.side]
        nearest = min(on_side, key=lambda r: (r.topological_dist, r.distance_bp), default=None)
        if t.effect == 'disrupted':
            score.disrupted_truth += 1
            score.disrupted_truth_found += bool(nearest and nearest.effect & DISRUPTING)
        if nearest is None:
            score.misses.add(t.key)
            continue
        score.found.add(t.key)
        score.bands.append((band(t.distance_bp), band(nearest.distance_bp, nearest.spatial in OVERLAP_SIDES)))
        # eris reports an overlap as distance 0
        score.distance_errors.append(abs(nearest.distance_bp - max(0, t.distance_bp)))
        if not any(sequence_agrees(p, t) for p in place(nearest.feature.id) if matches(p, t)):
            score.nucleotide_mismatches.add(t.key)

    if reach_loci is not None:
        reach_flanks = [f for locus in reach_loci for f in read_flanks(locus, groups)]
        score.budget_misses = {t.key for t in truths if t.key in score.misses
                               and any(f.side == t.side for f in matching_flanks(t, reach_flanks, place))}
    return score


def headline(score: RelationScore) -> list[str]:
    """Recall on each side and pooled, and precision of first flanks."""
    found = {side: sum(k[1] == side for k in score.found) for side in SIDES}
    return [
        *(f"recall {side}: {found[side]}/{score.total[side]} = {score.recall[side]:.3f}" for side in SIDES),
        f"recall pooled over sides: {len(score.found_any_side)}/{sum(score.total.values())} = {score.pooled_recall:.3f}",
        f"precision of first flanks: {score.true_first_flanks}/{score.first_flanks} = {score.precision:.3f}",
    ]


def report(score: RelationScore, by_family: RelationScore | None = None) -> str:
    """A readable summary of a score, for the test log; `by_family` is the same run scored per IS family."""
    lines = [
        *headline(score),
        f"distance band agreement: {score.band_agreement:.3f} (median error {score.median_distance_error} bp, "
        f"max {score.max_distance_error} bp)",
        f"disrupted precision: {score.disrupted_called_true}/{score.disrupted_called} = {score.disrupted_precision:.3f}",
        f"disrupted recall: {score.disrupted_truth_found}/{score.disrupted_truth} = {score.disrupted_recall:.3f}",
        f"misses: {len(score.misses)}, of which {len(score.budget_misses)} within reach of a larger hops budget",
        *(f"  {cause:6} {k}" for k, cause in sorted(score.miss_causes.items())),
        f"nucleotide mismatches: {sorted(score.nucleotide_mismatches)}",
        f"UPREGULATED calls (not scored): {score.upregulated}",
        f"unplaceable flank genes: {score.unplaceable}",
    ]
    if by_family is not None:
        lines += ["per IS family, for reference:", *(f"  {line}" for line in headline(by_family))]
    return "\n".join(lines)


# Assignment score (#52) ---------------------------------------------------------------------------------------------
TIERS = ('side', 'nearest', 'distance')
AssignmentKey = tuple[str, str, str]  # (isoform group, side, eris gene id)
FP_CAUSES = ('FP_unplaced', 'FP_overcall', 'FP_side', 'FP_group', 'FP_not_nearest')
FN_CAUSES = ('FN_undetected', 'FN_side', 'FN_group', 'FN_budget', 'FN_unreported')


@dataclass(frozen=True)
class ISCopy:
    """A truth IS copy from the BED."""
    name: str
    group: str
    replicon: str
    start: int  # 0-based, half-open
    end: int
    strand: str


@dataclass(frozen=True)
class ExpectedRelation:
    """
    The nearest eris gene on one side of the truth copies of one isoform group. `genes` holds every eris gene id that
    lands at that place (the same gene called twice, or an exact tie); reporting any of them is right.
    """
    group: str
    side: str
    genes: frozenset[str]
    distances: frozenset[int]  # one per copy it flanks; negative when the gene crosses the IS's end
    copies: frozenset[ISCopy]

    @property
    def key(self) -> AssignmentKey:
        return self.group, self.side, min(self.genes)

    def distance_band_agrees(self, relation: FeatureRelation) -> bool:
        """Whether eris's distance falls in the band of any copy's, since eris does not say which copy it flanks."""
        return _eris_band(relation) in {band(d) for d in self.distances}

    def distance_error(self, relation: FeatureRelation) -> int:
        """|eris - reference| bp to the nearest of the copies' distances, with overlaps read as 0 as eris reports them."""
        return min(abs(relation.distance_bp - max(0, d)) for d in self.distances)


def read_is_copies(path: Path, groups: dict[str, str], replicons: set[str] | None = None) -> list[ISCopy]:
    """Reads the truth IS copies of a BED (replicon, start, end, name, score, strand), optionally on some replicons."""
    copies = []
    with open(path) as handle:
        for line in handle:
            replicon, start, end, name, _, strand = line.rstrip('\n').split('\t')[:6]
            if replicons is None or replicon in replicons:
                copies.append(ISCopy(name, group_of(name, groups), replicon, int(start), int(end), strand))
    return copies


def _inside(p: Placement, c: ISCopy) -> bool:
    return p.replicon == c.replicon and c.start <= p.start and p.end <= c.end


def _half_overlap(a: Placement | ISCopy, b: Placement | ISCopy) -> bool:
    """Same replicon, overlapping by at least half of the shorter, as `matches` asks of a gene and a CDS."""
    shorter = min(a.end - a.start, b.end - b.start)
    return a.replicon == b.replicon and min(a.end, b.end) - max(a.start, b.start) >= MIN_OVERLAP * shorter


def _same_gene(p: Placement, q: Placement) -> bool:
    return p.strand == q.strand and _half_overlap(p, q)


def side_distances(c: ISCopy, p: Placement, lengths: dict[str, int] | None = None) -> dict[str, int]:
    """
    The distance from a copy to a gene on each side the gene lies on, relative to the IS strand; negative when the gene
    crosses that end of the IS. A replicon in `lengths` is circular, so a gene can lie across its origin.
    """
    if p.replicon != c.replicon:
        return {}
    length = (lengths or {}).get(c.replicon)
    left, right = [], []
    for shift in (0, -length, length) if length else (0,):
        start, end = p.start + shift, p.end + shift
        if start < c.start:
            left.append(c.start - end)
        if end > c.end:
            right.append(start - c.end)
    up, down = (left, right) if c.strand == '+' else (right, left)
    return {side: min(ds) for side, ds in zip(SIDES, (up, down)) if ds}


def _universe(copies: list[ISCopy], genes: Iterable[str],
              place: Callable[[str], list[Placement]]) -> dict[str, list[Placement]]:
    """eris's placed genes, at their places outside every truth IS."""
    universe = {}
    for gene in genes:
        if placements := [p for p in place(gene) if not any(_inside(p, c) for c in copies)]:
            universe[gene] = placements
    return universe


def expected_relations(copies: list[ISCopy], genes: Iterable[str], place: Callable[[str], list[Placement]],
                       lengths: dict[str, int] | None = None) -> list[ExpectedRelation]:
    """The nearest of eris's genes on each side of each truth copy, merged per (group, side, gene)."""
    return _expected(copies, _universe(copies, genes, place), lengths)


def _expected(copies: list[ISCopy], universe: dict[str, list[Placement]],
              lengths: dict[str, int] | None) -> list[ExpectedRelation]:
    by_key: dict[AssignmentKey, ExpectedRelation] = {}
    for c in copies:
        for side in SIDES:
            candidates = [(d, gene, p) for gene, placements in universe.items() for p in placements
                          if (d := side_distances(c, p, lengths).get(side)) is not None]
            if not candidates:
                continue
            nearest = min(d for d, _, _ in candidates)
            places = [p for d, _, p in candidates if d == nearest]
            found = frozenset(g for d, g, p in candidates if d == nearest or any(_same_gene(p, q) for q in places))
            relation = ExpectedRelation(c.group, side, found, frozenset({nearest}), frozenset({c}))
            if (old := by_key.get(relation.key)) is not None:
                relation = ExpectedRelation(c.group, side, old.genes | found, old.distances | relation.distances,
                                            old.copies | relation.copies)
            by_key[relation.key] = relation
    return list(by_key.values())


@dataclass
class AssignmentScore:
    """
    How eris assigns its own gene calls to IS sides. The headline leaves out IS-detection errors: first flanks of a
    group with no truth copy (FP_overcall), and expected relations of copies no locus of the group covers
    (FN_undetected).
    """
    first_flanks: int = 0
    overcalls: int = 0
    precise: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TIERS, 0))  # true first flanks per tier
    expected: int = 0
    undetected: int = 0
    recalled: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TIERS, 0))  # found relations per tier
    band_agreements: list[bool] = field(default_factory=list)  # one per found relation
    distance_errors: list[int] = field(default_factory=list)
    errors: set[tuple[AssignmentKey, str]] = field(default_factory=set)

    def precision(self, tier: str) -> float:
        n = self.first_flanks - self.overcalls
        return self.precise[tier] / n if n else 0.0

    def recall(self, tier: str) -> float:
        n = self.expected - self.undetected
        return self.recalled[tier] / n if n else 0.0

    @property
    def band_agreement(self) -> float:
        return sum(self.band_agreements) / len(self.band_agreements) if self.band_agreements else 0.0

    def causes(self) -> dict[str, int]:
        return {cause: sum(c == cause for _, c in self.errors) for cause in FP_CAUSES + FN_CAUSES}


def _eris_band(relation: FeatureRelation) -> str:
    return band(relation.distance_bp, relation.spatial in OVERLAP_SIDES)


def score_assignment(copies: list[ISCopy], genes: Iterable[str], loci: list, groups: dict[str, str],
                     place: Callable[[str], list[Placement]], cover: Callable[[object], list[Placement]],
                     reach_loci: list | None = None, lengths: dict[str, int] | None = None) -> AssignmentScore:
    """
    Scores how eris assigns its own genes to IS sides (#52). `genes` are the ids of every gene eris called, `place`
    returns a gene's reference placements, and `cover` the reference placements of a locus's IS segments.
    `reach_loci` are the loci of a run with a larger hops budget, to tell budget misses apart.
    """
    universe = _universe(copies, genes, place)
    expected = _expected(copies, universe, lengths)
    by_gene: dict[str, list[ExpectedRelation]] = {}
    for e in expected:
        for gene in e.genes:
            by_gene.setdefault(gene, []).append(e)
    truth_groups = {c.group for c in copies}
    per_locus = [(locus, read_flanks(locus, groups)) for locus in loci]
    score = AssignmentScore(expected=len(expected))

    for f in (f for _, flanks in per_locus for f in first_flanks(flanks)):
        gene = f.relation.feature.id
        score.first_flanks += 1
        key = (f.group, f.side, gene)
        if not place(gene):
            score.errors.add((key, 'FP_unplaced'))
            continue
        if f.group not in truth_groups:
            score.overcalls += 1
            score.errors.add((key, 'FP_overcall'))
            continue
        nearest = {(e.group, e.side): e for e in by_gene.get(gene, [])}
        distances: dict[str, int] = {}
        for c in (c for c in copies if c.group == f.group):
            for p in universe.get(gene, []):
                for side, d in side_distances(c, p, lengths).items():
                    distances[side] = min(d, distances.get(side, d))
        expected_here = nearest.get((f.group, f.side))
        # The side tier contains the nearest tier: a gene between two copies can be nearest to the farther one
        on_nearer_side = f.side in distances and distances[f.side] == min(distances.values())
        score.precise['side'] += expected_here is not None or on_nearer_side
        if expected_here is not None:
            score.precise['nearest'] += 1
            score.precise['distance'] += expected_here.distance_band_agrees(f.relation)
        elif any(group == f.group for group, _ in nearest):
            score.errors.add((key, 'FP_side'))
        elif nearest:
            score.errors.add((key, 'FP_group'))
        else:
            score.errors.add((key, 'FP_not_nearest'))

    reported = [(locus_group(locus, groups), f) for locus, flanks in per_locus for f in flanks]
    reach_flanks = [f for locus in reach_loci or [] for f in read_flanks(locus, groups)]
    for e in expected:
        of_group = [f for group, f in reported if group == e.group and f.relation.feature.id in e.genes]
        on_side = [f.relation for f in of_group if f.side == e.side]
        if on_side:
            r = min(on_side, key=lambda r: (r.topological_dist, r.distance_bp))
            score.recalled['side'] += 1
            score.recalled['nearest'] += 1
            score.recalled['distance'] += e.distance_band_agrees(r)
            score.band_agreements.append(e.distance_band_agrees(r))
            score.distance_errors.append(e.distance_error(r))
            continue
        if not any(_half_overlap(p, c) for locus, _ in per_locus if locus_group(locus, groups) == e.group
                   for p in cover(locus) for c in e.copies):
            score.undetected += 1
            cause = 'FN_undetected'
        elif of_group:
            cause = 'FN_side'
        elif any(f.relation.feature.id in e.genes for _, f in reported):
            cause = 'FN_group'
        elif any(f.group == e.group and f.side == e.side and f.relation.feature.id in e.genes for f in reach_flanks):
            cause = 'FN_budget'
        else:
            cause = 'FN_unreported'
        score.errors.add((e.key, cause))
    return score


def assignment_report(score: AssignmentScore) -> str:
    """The three tiers, the errors by cause, and each error, for the test log."""
    first, expected = score.first_flanks - score.overcalls, score.expected - score.undetected
    labels = {'side': 'side', 'nearest': 'side + nearest (headline)', 'distance': 'side + nearest + distance band'}
    errors = sorted(score.errors, key=lambda e: (e[1], e[0]))
    causes = score.causes()
    return "\n".join([
        "assignment of eris's own genes to IS sides (IS-detection errors excluded):",
        *(f"  {labels[t]:32} precision {score.precise[t]}/{first} = {score.precision(t):.3f}  "
          f"recall {score.recalled[t]}/{expected} = {score.recall(t):.3f}" for t in TIERS),
        f"  distance on found relations: band agreement {score.band_agreement:.3f}, error median "
        f"{median(score.distance_errors) if score.distance_errors else 0} bp, "
        f"max {max(score.distance_errors, default=0)} bp",
        "  (recall is the same in the side and nearest tiers: every expected relation is a nearest gene)",
        f"  excluded: FP_overcall {score.overcalls} of {score.first_flanks} first flanks, "
        f"FN_undetected {score.undetected} of {score.expected} expected relations",
        "  FP by cause: " + ", ".join(f"{c} {causes[c]}" for c in FP_CAUSES),
        "  FN by cause: " + ", ".join(f"{c} {causes[c]}" for c in FN_CAUSES),
        *(f"    {cause:15} {key}" for key, cause in errors),
    ])
