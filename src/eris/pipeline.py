from dataclasses import dataclass, field
from typing import Iterable, Optional, Any
from threading import local as thread_local
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np

from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation, SeqFeature, CompoundLocation

from mappy import ThreadBuffer

from pyfgs import GeneFinder, Model, Gene, Mutation

from eris.io import TargetDatabase, GenomeAssembly, gene_id
from eris.graph import TopologyEngine, Edge, Branch
from eris.alignment import AlignmentBatch, AlignmentRecord, pooled_identity
from eris.interval import IntervalBatch, Context, Strand, Interval
from eris.constants import FeatureType, Orientation, Effect, DEFAULT_MAX_FLANK_DISTANCE
from eris.copy_number import CopyNumber, assign_copy_numbers, covers_contig


# DataClasses ----------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class LocationSegment:
    """A single continuous segment of a genomic location (e.g. an exon)."""
    contig: str
    start: int
    end: int
    strand: Strand

    def to_biopython(self) -> SimpleLocation:
        """Convert to a Biopython SimpleLocation object."""
        return SimpleLocation(self.start, self.end, self.strand)


@dataclass(slots=True)
class GenomicFeature:
    """
    Represents a biological feature (e.g. a gene, MGE) which may span multiple segments.

    Example:
        >>> seg = LocationSegment("ctg1", 100, 500, Strand.FORWARD)
        >>> feat = GenomicFeature("geneA", FeatureType.CDS, [seg])
        >>> feat.is_multi_contig
        False
    """
    id: str
    type: FeatureType
    segments: list[LocationSegment]
    qualifiers: dict[str, list[str]] = field(default_factory=dict)
    identity: Optional[float] = None  # Target identity, set only on target features

    @property
    def is_multi_contig(self) -> bool:
        """True if the feature spans more than one contig."""
        return len(set(seg.contig for seg in self.segments)) > 1

    def to_biopython(self) -> SeqFeature:
        """Convert to a Biopython SeqFeature object."""
        if len(self.segments) == 1:
            loc = self.segments[0].to_biopython()
        else:
            loc = CompoundLocation([s.to_biopython() for s in self.segments])
        return SeqFeature(location=loc, type=self.type.value, id=self.id, qualifiers=self.qualifiers)

    @property
    def bounding_start(self) -> int:
        """The absolute minimum start coordinate of all segments."""
        return self.segments[0].start

    @property
    def bounding_end(self) -> int:
        """The absolute maximum end coordinate of all segments."""
        return self.segments[-1].end

    @property
    def bounding_strand(self) -> Strand:
        """The strand of the primary (first) segment."""
        return self.segments[0].strand


@dataclass(slots=True)
class FeatureRelation:
    """
    Describes the topological relationship between a target and a contextual feature.

    Attributes:
        feature: The contextual (passenger or flanking) GenomicFeature.
        spatial: The spatial Context (e.g. UPSTREAM).
        distance_bp: Physical distance in base pairs.
        topological_dist: Distance in graph hops.
        orientation: Relative orientation (SAME or OPPOSITE).
        effect: Combined Effect flags (e.g. TRUNCATED).
    """
    feature: GenomicFeature
    spatial: Context
    distance_bp: int
    topological_dist: int
    orientation: Orientation
    effect: Effect = Effect.NONE

    def get_relative_position(self, locus_start: int, locus_end: int) -> str:
        """Returns a string describing the relative position to a locus boundary."""
        g_start = self.feature.bounding_start
        g_end = self.feature.bounding_end
        g_strand = self.feature.bounding_strand

        if locus_end <= g_start:
            return "5_prime_flank" if g_strand == Strand.FORWARD else "3_prime_flank"
        elif locus_start >= g_end:
            return "3_prime_flank" if g_strand == Strand.FORWARD else "5_prime_flank"
        return "overlapping"


@dataclass(slots=True)
class Locus:
    """
    A single assembled genomic region containing a target of interest and its context.

    The locus may be a simple interval on one contig or a complex 'stitched' path
    across multiple contigs in an assembly graph.
    """
    id: str
    contig: str
    start: int
    end: int
    targets: list[GenomicFeature]
    passengers: list['FeatureRelation']
    upstream_flanks: list['FeatureRelation']
    downstream_flanks: list['FeatureRelation']
    fractional_depth: Optional[float] = 1.0  # Fractional depth in variant mode, copy number in collapse; None: no depth
    copy_number: Optional[CopyNumber] = None  # Collapse mode only

    def extract_sequence(self, genome: 'GenomeAssembly') -> str:
        """
        Extracts the nucleotide sequence of the locus from the assembly.

        A local locus is its slice of the contig. A stitched locus follows its contig path: each contig on the strand
        it is read on, the edge overlap with the contig before it counted once, cut to the ends of the target on the
        first and last contigs.
        """
        if len(segments := self.targets[0].segments) == 1:
            return str(genome[self.contig][self.start:self.end])

        parts, prev = [], None
        for seg in segments:
            seq = genome[seg.contig]
            if seg.strand == Strand.REVERSE:
                seq = seq.reverse_complement()
            skip = genome.overlap(prev.contig, prev.strand, seg.contig, seg.strand) if prev else 0
            parts.append(str(seq[skip:]))
            prev = seg
        path = "".join(parts)

        # Bases before the target on the first contig and after it on the last, as each is read
        first, last = segments[0], segments[-1]
        head = first.start if first.strand != Strand.REVERSE else len(genome[first.contig]) - first.end
        tail = len(genome[last.contig]) - last.end if last.strand != Strand.REVERSE else last.start
        return path[head:len(path) - tail]

# Constants ------------------------------------------------------------------------------------------------------------
MIN_SHARED_FRACTION = 0.5  # Of the shorter alignment, for two alignments to be one IS copy
BRANCH_ENTRY_SLACK = 10  # bp into a hub branch's first contig a copy's target may start and still run on from the hub


# Functions ------------------------------------------------------------------------------------------------------------
def _shares_half(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Whether two ranges share at least half the shorter one's bases."""
    shared = min(a_end, b_end) - max(a_start, b_start)
    return shared >= MIN_SHARED_FRACTION * min(a_end - a_start, b_end - b_start)


def same_copy_groups(alignments: 'AlignmentBatch', tolerance: int = 0) -> list[np.ndarray]:
    """
    Groups the alignments on one contig by the IS copy they lie on (#57, docs/adr/0006).

    Two alignments are one copy when they share at least half the shorter one's bases, as near-identical targets
    (IS1A, IS1R) aligning to one copy do, or when they are two pieces of one alignment: the same target on the same
    strand, touching or overlapping on the contig, and following on in the target in that order, as when a deletion in
    the contig splits it. Copies that merely touch or overlap by a few bases stay apart, even copies of the same target.
    Grouping is transitive.

    Args:
        alignments: The alignments on one contig.
        tolerance: Above 0, alignments up to this many bp apart on the contig are also one group.

    Returns:
        Each group's indices into `alignments`, in contig order, groups ordered by their first alignment.
    """
    order = np.lexsort((alignments.t_ends, alignments.t_starts))
    t_s, t_e = alignments.t_starts[order], alignments.t_ends[order]
    q_s, q_e = alignments.q_starts[order], alignments.q_ends[order]
    names, strands = alignments.q_names[order], alignments.strands[order]
    n = len(order)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def pieces_of_one_alignment(i: int, j: int) -> bool:
        """j starts at or after i on the contig and touches it; do they follow on in the target, in strand order?"""
        if names[i] != names[j] or strands[i] != strands[j] or t_s[j] > t_e[i]:
            return False
        if _shares_half(q_s[i], q_e[i], q_s[j], q_e[j]):
            return False  # each covers the same part of the target, so two copies
        return bool(q_s[j] > q_s[i]) if strands[i] != Strand.REVERSE else bool(q_s[j] < q_s[i])

    for i in range(n):
        for j in range(i + 1, n):
            if t_s[j] > t_e[i] + tolerance:
                break  # sorted by start, so no later alignment reaches i either
            if (_shares_half(t_s[i], t_e[i], t_s[j], t_e[j]) or pieces_of_one_alignment(i, j)
                    or (tolerance > 0 and t_s[j] - t_e[i] <= tolerance)):
                parent[root(j)] = root(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [order[g] for g in groups.values()]


# Classes --------------------------------------------------------------------------------------------------------------
class LocusBuilder:
    """
    Orchestrates the assembly of loci from raw alignments and genomic context.

    It uses the TopologyEngine to resolve graph-spanning alignments and then
    identifies flanking and passenger genes for each identified locus.
    """
    __slots__ = ('topology_engine', 'genome', 'target_feature_type', 'max_feature_hops',
                 'locus_tolerance', 'features', 'genes', 'max_flank_distance', 'alignments')

    def __init__(self, topology_engine: 'TopologyEngine', genome: 'GenomeAssembly',
                 target_feature_type: FeatureType = FeatureType.CDS,
                 max_feature_hops: int = 3, locus_tolerance: int = 0,
                 features: dict[str, list[GenomicFeature]] = None,
                 genes: dict[str, list[Gene]] = None,
                 max_flank_distance: Optional[int] = DEFAULT_MAX_FLANK_DISTANCE):
        """
        Initialize the LocusBuilder.

        Args:
            topology_engine: Engine for graph traversal.
            genome: The full assembly and metadata.
            target_feature_type: Classification for primary alignment targets.
            max_feature_hops: Max contextual genes to look for in each direction.
            locus_tolerance: Alignments up to this many bp apart share a locus; at 0, only those on one IS copy do.
            features: Dictionary of GenomicFeatures per contig.
            genes: Dictionary of PyFGS Gene objects per contig.
            max_flank_distance: Max distance (bp) from the target to a flank, on its own contig or across the
                graph; None for no limit.
        """
        self.topology_engine = topology_engine
        self.genome = genome
        self.target_feature_type = target_feature_type
        self.max_feature_hops = max_feature_hops
        self.locus_tolerance = locus_tolerance
        self.features = features or {}
        self.genes = genes or {}
        self.max_flank_distance = max_flank_distance
        self.alignments: dict[str, AlignmentBatch] = {}

    def assemble(self, alignments: dict, min_coverage: float = 0.30, min_identity: float = 0.0) -> Iterable['Locus']:
        """
        The main entry point for generating loci.

        Stitches graph-spanning alignments and processes local alignments
        to produce a sequence of Locus objects. Filtering happens after stitching,
        so a low-identity fragment can still join a stitched path; the path is then
        kept or dropped on its pooled target identity.
        """

        # The real alignments, before stitching adds synthetic records: a hub's branches are measured from them (#65)
        self.alignments = alignments
        # Now expects a list of paths (lists of AlignmentRecords) instead of pairs
        cleaned_alignments, resolved_paths = self.topology_engine.resolve_split_alignments(alignments)

        # Filter and yield STITCHED paths
        kept_paths = []
        for path in resolved_paths:
            # Calculate total coverage of the stitched path against the IS query
            coverage = (path[-1].q_end - path[0].q_start) / path[0].q_length
            if coverage >= min_coverage and pooled_identity(path) >= min_identity:
                kept_paths.append(path)
        if getattr(self.topology_engine, 'mode', 'variant') == 'collapse':
            kept_paths = self._one_path_per_hub(kept_paths)
        for path in kept_paths:
            yield self._stitch(path)

        # Filter and yield LOCAL (unstitched) alignments
        for contig_id, batch in cleaned_alignments.items():
            cov_mask = ((batch.q_ends - batch.q_starts) / batch.q_lengths) >= min_coverage
            id_mask = batch.identities >= min_identity
            intact_batch = batch.filter(cov_mask & id_mask)

            if len(intact_batch) > 0:
                contig_gene_intervals = self.topology_engine.features.get(contig_id, IntervalBatch.empty())
                for locus in self._build_local(contig_id, intact_batch, contig_gene_intervals):
                    yield locus

    def _one_path_per_hub(self, paths: list[list['AlignmentRecord']]) -> list[list['AlignmentRecord']]:
        """
        Keeps one stitched path per repeat hub (#14). Near-identical targets (IS1A, IS1R, ...) each stitch their own
        path through the contigs one collapsed repeat covers end to end, mostly as synthetic records, and would each
        report the whole hub. Paths that share a hub contig, directly or through another path, are one locus; the
        one that aligns the most target bases names it.
        """
        overlap = max((e.overlap for e in self.genome.edges), default=0)
        groups: list[tuple[set[str], list[list['AlignmentRecord']]]] = []
        for path in paths:
            hub = {f.t_name for f in path if covers_contig(f.t_length, f.t_start, f.t_end, overlap)}
            members = [path]
            for group in [g for g in groups if g[0] & hub]:
                other_hub, other_members = group
                hub |= other_hub
                members = other_members + members
                groups.remove(group)
            groups.append((hub, members))

        def aligned_bases(path: list['AlignmentRecord']) -> tuple[int, float]:
            """Target bases the path's real fragments align, then pooled identity to break ties."""
            return sum(f.q_end - f.q_start for f in path if f.idx != -1), pooled_identity(path)
        return [max(members, key=aligned_bases) for _, members in groups]

    def _hub_branch_links(self, fragments: list['AlignmentRecord']) -> tuple[list[Edge], list[Edge]]:
        """
        In collapse mode, the links leaving a stitched path's repeat hub upstream and downstream, other than those the
        flank walk from the path's own end contig already follows (#56). A hub reported once stands for every copy
        collapsed into it, so each of its links is a branch with its own flanks. Only the links leaving the hub's
        outermost contig on each side count: a link out of the middle of the hub enters a stub of one copy's variant
        of the target, which brings more partial gene calls than flanks. Variant mode has none: each path through a
        hub is its own locus.
        """
        if getattr(self.topology_engine, 'mode', 'variant') != 'collapse':
            return [], []
        overlap = max((e.overlap for e in self.genome.edges), default=0)
        hub: dict[str, Strand] = {}
        for f in fragments:
            if covers_contig(f.t_length, f.t_start, f.t_end, overlap):
                hub.setdefault(f.t_name, Strand(f.strand))
        if not hub:
            return [], []
        in_order = list(hub)  # fragments are in the target's direction, so the first hub contig is upstream
        links_by_side: list[list[Edge]] = []
        for side, end, outer in ((-1, fragments[0], in_order[0]), (1, fragments[-1], in_order[-1])):
            if end.t_name == outer:  # the walk starts on the hub's edge, so it already leaves by every link
                links_by_side.append([])
                continue
            # The walk from the end contig is the branch along the link into it
            links_by_side.append([e for e in self.topology_engine.links_leaving(hub, side)
                                  if e.u == outer and (e.v, e.v_strand) != (end.t_name, end.strand * side)])
        return links_by_side[0], links_by_side[1]

    def _target_past_link(self, e: Edge) -> int:
        """
        How far a copy's target runs onto a hub branch's first contig `e.v` past the overlap it shares with the hub
        (#65), read along the walk: the furthest end of a real alignment that starts where the contig is entered,
        within 10 bp. 0 if none does. The hub's edge is target end to end, so a copy carrying on past it starts in the
        overlap; an alignment starting past it is another IS beside this one. Synthetic stitch records are left out:
        they overshoot by one overlap.
        """
        batch = self.alignments.get(e.v)
        if batch is None or not len(batch):
            return 0
        if e.v_strand == Strand.FORWARD:
            starts, ends = batch.t_starts, batch.t_ends
        else:
            length = self.genome.contig_lengths[e.v]
            starts, ends = length - batch.t_ends, length - batch.t_starts
        entering = starts <= BRANCH_ENTRY_SLACK
        if not entering.any():
            return 0
        return max(0, int(ends[entering].max()) - e.overlap)

    def _resolve_relation(self, contig: str, idx: int, interval_batch: 'IntervalBatch',
                          spatial: Context, dist: int, topo: int, target_strand: Strand,
                          target_bounds: tuple[int, int]) -> FeatureRelation:
        """Analyzes and creates a FeatureRelation for a specific gene-target pair."""

        orig_idx = interval_batch.original_indices[idx]  # type: int
        feature = self.features[contig][orig_idx]  # type: GenomicFeature
        raw_pyfgs_gene = self.genes[contig][orig_idx]  # type: Gene
        gene_strand = Strand(interval_batch.strands[idx])

        # Determine strict relational spatial context natively using coordinate limits for local genes
        if topo == 0:
            target_interval = Interval(target_bounds[0], target_bounds[1], target_strand)
            gene_interval = Interval(interval_batch.starts[idx], interval_batch.ends[idx], gene_strand)
            spatial = target_interval.relate(gene_interval)

        effect = Effect.NONE

        if spatial in (Context.OVERLAPPING, Context.OVERLAPPING_START, Context.OVERLAPPING_END):
            effect = Effect.DISRUPTED

        # If the gene was biologically broken by the insertion. `target_bounds` are on the target's own contig, so a
        # gene reached across a link (topo > 0) cannot be checked against them (#64).
        if topo == 0 and (raw_pyfgs_gene.insertions or raw_pyfgs_gene.deletions):
            # Dynamically fetch the sequence from the genome for the mutation checker
            for mut in raw_pyfgs_gene.mutations(bytes(self.genome[contig])):  # type: Mutation
                dist_to_start = abs(mut.pos - target_bounds[0])
                dist_to_end = abs(mut.pos - target_bounds[1])

                if dist_to_start <= 5 or dist_to_end <= 5:
                    effect = Effect.TRUNCATED if effect == Effect.NONE else effect | Effect.TRUNCATED
                    break

        if target_strand == Strand.UNSTRANDED or gene_strand == Strand.UNSTRANDED:
            orientation = Orientation.NONE
        elif target_strand == gene_strand:
            orientation = Orientation.SAME
        else:
            orientation = Orientation.OPPOSITE

        return FeatureRelation(
            feature=feature, spatial=spatial, distance_bp=dist,
            topological_dist=topo, orientation=orientation, effect=effect
        )

    @staticmethod
    def _before_target_across_origin(intervals: 'IntervalBatch', original_index: int, walk_direction: int,
                                     target_bounds: tuple[int, int]) -> bool:
        """
        Whether a gene that a walk reaches back on its own contig, across the origin of a circular contig, lies
        wholly before the target in the walk's direction. Going any further round the circle would reach the target
        itself. On a circle with few genes, a gene can still flank the target on both sides.
        """
        j = int(np.flatnonzero(intervals.original_indices == original_index)[0])
        if walk_direction == 1:
            return bool(intervals.ends[j] <= target_bounds[0])
        return bool(intervals.starts[j] >= target_bounds[1])

    def _extract_flanks(self, contig: str, boundary: int, walk_direction: int,
                        context: Context, intervals: Optional['IntervalBatch'], dest_list: list,
                        target_strand: Strand, target_bounds: tuple[int, int], claimed: set[tuple[str, int]],
                        branch_links: Iterable[Edge] = (), blocked: frozenset[str] = frozenset()):
        """
        Walks the contig (and graph) from a boundary to find flanking features. `claimed` holds the genes the graph
        walks have already reported for this locus (see TopologyEngine.traverse); start a new one for each locus.
        Features further than `max_flank_distance` from the boundary are left out, wherever they lie (#55).
        `branch_links` are the other links leaving a repeat hub on this side, each walked as its own branch with the
        full flank budget (#56); flanks along them are measured from where that branch's own copy of the target ends:
        the hub's edge, where the link leaves it, plus any target the copy carries onto the branch's first contig (#65).
        The graph walk never enters a `blocked` contig. A gene across a link that runs back over the target's end is
        0 bp away and overlaps the target's start (upstream) or end (downstream), as a local gene would (#64).
        """
        rem_hops = self.max_feature_hops
        exit_strand = Strand.REVERSE if walk_direction == -1 else Strand.FORWARD
        limit = self.max_flank_distance

        # 1. Local Search
        if intervals:
            if walk_direction == -1:
                idx = np.searchsorted(intervals.ends, boundary, side='right')
                for i in reversed(range(max(0, idx - self.max_feature_hops), idx)):
                    d = max(0, boundary - intervals.ends[i])
                    if limit is not None and d > limit: continue
                    f = self._resolve_relation(contig, i, intervals, context, int(d), 0, target_strand, target_bounds)
                    dest_list.append(f)
                    rem_hops -= 1
            else:
                idx = np.searchsorted(intervals.starts, boundary, side='left')
                for i in range(idx, min(len(intervals), idx + self.max_feature_hops)):
                    d = max(0, intervals.starts[i] - boundary)
                    if limit is not None and d > limit: continue
                    f = self._resolve_relation(contig, i, intervals, context, int(d), 0, target_strand, target_bounds)
                    dest_list.append(f)
                    rem_hops -= 1

        # 2. Graph Spillover
        # The walk reads the contig along its direction from 0, so the boundary sits here in its coordinates
        walk_boundary = boundary if walk_direction == 1 else self.genome.contig_lengths[contig] - boundary
        # The link's source contig, in the hub, is placed to end at the boundary, then pulled back by the target its
        # copy carries onto the next contig, so a flank across the link is measured from that copy's end of the target
        branches = [Branch(e, self.max_feature_hops,
                           walk_boundary - self.genome.contig_lengths[e.u] - self._target_past_link(e))
                    for e in branch_links]
        if rem_hops > 0 or branches:
            reach = None if limit is None else walk_boundary + limit
            for s_ctg, node_depth, batch in self.topology_engine.traverse(contig, exit_strand, rem_hops, claimed,
                                                                          reach, branches, blocked):
                across_origin = s_ctg == contig and contig in self.topology_engine.circular
                for i in range(len(batch)):
                    if across_origin and not self._before_target_across_origin(
                            self.topology_engine.features[contig], batch.original_indices[i], walk_direction,
                            target_bounds):
                        continue
                    # The traversal counts each link's overlap once, so a negative gap is a gene that runs back
                    # across the target's end, such as one starting in the overlap of the next contig (#64)
                    d = batch.starts[i] - walk_boundary
                    spatial = context
                    if d < 0:
                        # Upstream crosses the target's start and downstream its end, in the target's direction, as
                        # Interval.relate gives a local gene
                        spatial = Context.OVERLAPPING_START if context == Context.UPSTREAM \
                            else Context.OVERLAPPING_END
                        d = 0
                    dest_list.append(
                        self._resolve_relation(s_ctg, i, batch, spatial, int(d), node_depth, target_strand,
                                               target_bounds)
                    )

    def _build_local(self, contig: str, alignment_batch: 'AlignmentBatch', gene_intervals: 'IntervalBatch') -> list[
        'Locus']:
        """
        Identifies loci within a single contig: one per IS copy its alignments lie on (see
        `same_copy_groups`), so two IS copies that only touch are two loci, each with its own strand and sides (#57).
        """
        loci = []
        for group in same_copy_groups(alignment_batch, self.locus_tolerance):
            targets = [alignment_batch.get_record(idx) for idx in group]
            span = Interval(int(alignment_batch.t_starts[group].min()), int(alignment_batch.t_ends[group].max()))

            target_features = [
                GenomicFeature(t.q_name, self.target_feature_type,
                               [LocationSegment(contig, t.t_start, t.t_end, Strand(t.strand))],
                               identity=t.identity)
                for t in targets
            ]

            # Collapse mode's copy number is set once every locus is built (see assign_copy_numbers)
            locus = Locus(
                id=f"locus_{uuid4().hex[:8]}", contig=contig, start=span.start, end=span.end,
                targets=target_features, passengers=[], upstream_flanks=[], downstream_flanks=[],
                fractional_depth=1.0 if self.genome.has_depth else None
            )

            # The locus is read on its first target's strand
            primary_strand = Strand(targets[0].strand)
            span_bounds = (span.start, span.end)

            internal_indices = gene_intervals.query(span.start, span.end)
            for idx in internal_indices:
                locus.passengers.append(
                    self._resolve_relation(contig, idx, gene_intervals, Context.INSIDE, 0, 0, primary_strand,
                                           span_bounds))

            if primary_strand == Strand.REVERSE:
                u_dir, u_bound = 1, span.end
                d_dir, d_bound = -1, span.start
            else:
                u_dir, u_bound = -1, span.start
                d_dir, d_bound = 1, span.end

            claimed: set[tuple[str, int]] = set()
            self._extract_flanks(contig, u_bound, u_dir, Context.UPSTREAM, gene_intervals, locus.upstream_flanks,
                                 primary_strand, span_bounds, claimed)
            self._extract_flanks(contig, d_bound, d_dir, Context.DOWNSTREAM, gene_intervals, locus.downstream_flanks,
                                 primary_strand, span_bounds, claimed)

            loci.append(locus)

        return loci

    def _stitch(self, fragments: list['AlignmentRecord']) -> 'Locus':
        """Stitches multiple fragments into a single multi-contig locus."""
        first = fragments[0]
        last = fragments[-1]

        if getattr(self.topology_engine, 'mode', 'variant') == 'collapse':
            metric_val = 1.0  # Copy number is set once every locus is built (see assign_copy_numbers)
        elif not self.genome.has_depth:
            metric_val = None
        else:
            # Calculate the fractional flow of the stitched path
            source_depth = self.genome.contig_depths.get(first.t_name, 1.0)
            bottleneck_depth = min(self.genome.contig_depths.get(f.t_name, 1.0) for f in fragments)
            metric_val = round(bottleneck_depth / source_depth, 3) if source_depth > 0 else 1.0

        # 1. Build the multi-segment target feature across ALL fragments
        segments = [LocationSegment(f.t_name, f.t_start, f.t_end, Strand(f.strand)) for f in fragments]
        target_feature = GenomicFeature(
            id=f"{first.q_name}_stitched",
            type=self.target_feature_type,
            segments=segments,
            identity=pooled_identity(fragments)
        )

        locus = Locus(
            id=f"locus_split_{uuid4().hex[:8]}",
            contig="|".join(f.t_name for f in fragments),
            start=first.t_start,
            end=last.t_end,
            targets=[target_feature], passengers=[],
            upstream_flanks=[], downstream_flanks=[],
            fractional_depth=metric_val  # Holds either frac_depth or copy_number; NEW: Assign it to the Locus
        )

        # 2. UPSTREAM FLANKS (Strictly from the first fragment)
        u_dir = -1 if first.strand == 1 else 1
        u_bound = first.t_start if first.strand == 1 else first.t_end
        u_ints = self.topology_engine.features.get(first.t_name)
        claimed: set[tuple[str, int]] = set()
        u_links, d_links = self._hub_branch_links(fragments)
        # Once a hub's branches are walked, none may loop back into the stitched path: its genes are passengers
        blocked = frozenset(f.t_name for f in fragments) if u_links or d_links else frozenset()
        self._extract_flanks(
            first.t_name, u_bound, u_dir, Context.UPSTREAM, u_ints, locus.upstream_flanks,
            Strand(first.strand), (first.t_start, first.t_end), claimed, u_links, blocked
        )

        # 3. DOWNSTREAM FLANKS (Strictly from the last fragment)
        v_dir = 1 if last.strand == 1 else -1
        v_bound = last.t_end if last.strand == 1 else last.t_start
        v_ints = self.topology_engine.features.get(last.t_name)
        self._extract_flanks(
            last.t_name, v_bound, v_dir, Context.DOWNSTREAM, v_ints, locus.downstream_flanks,
            Strand(last.strand), (last.t_start, last.t_end), claimed, d_links, blocked
        )

        # 4. INSIDE PASSENGERS (Sweep across ALL fragments, including unaligned synthetic bubbles)
        for frag in fragments:
            f_ints = self.topology_engine.features.get(frag.t_name)
            if f_ints:
                internal_indices = f_ints.query(frag.t_start, frag.t_end)
                for idx in internal_indices:
                    locus.passengers.append(
                        self._resolve_relation(
                            frag.t_name, idx, f_ints, Context.INSIDE, 0, 0,
                            Strand(frag.strand), (frag.t_start, frag.t_end)
                        )
                    )

        return locus


class Pipeline:
    """
    High-level eris pipeline manager.

    Orchestrates the entire workflow: contig processing, target mapping,
    gene calling, graph building, and locus assembly. Uses a thread pool
    for parallel contig processing.
    """
    __slots__ = ('target_db', '_gene_finder', 'max_feature_hops', 'locus_tolerance', '_executor', 'mode', 'min_coverage',
                 'min_identity', 'max_flank_distance')
    _THREAD_LOCAL = thread_local()

    def __init__(self, target_db: 'TargetDatabase', max_feature_hops: int = 3, locus_tolerance: int = 0,
                 max_workers: Optional[int] = None, mode: str = 'variant', min_coverage: float = 0.30,
                 min_identity: float = 0.0, max_flank_distance: Optional[int] = DEFAULT_MAX_FLANK_DISTANCE):
        """
        Initialize the Pipeline.

        Args:
            target_db: Database of mapping targets (e.g. antibiotic resistance genes).
            max_feature_hops: Contextual search depth.
            locus_tolerance: Merging tolerance for adjacent hits.
            max_workers: Number of threads for parallel processing.
            mode: Resolution mode, 'variant' or 'collapse'.
            min_coverage: Minimum target coverage for a locus target to be reported.
            min_identity: Minimum target identity for a locus target to be reported.
            max_flank_distance: Max distance (bp) from a target to a flank, on its own contig or across the graph;
                None for no limit.
        """
        self.target_db = target_db
        self.max_feature_hops = max_feature_hops
        self.locus_tolerance = locus_tolerance
        self.mode = mode
        self.min_coverage = min_coverage
        self.min_identity = min_identity
        self.max_flank_distance = max_flank_distance
        self._gene_finder = GeneFinder(model=Model.Complete, whole_genome=False)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(cancel_futures=True, wait=False)
        self._gene_finder = None

    def _process_contig(self, contig: tuple[str, Seq]) -> tuple[
        str, Optional[IntervalBatch], Optional[AlignmentBatch], list[GenomicFeature], list[Gene]]:
        """Worker function for processing a single contig (Mapping + Gene Calling)."""

        if not hasattr(self._THREAD_LOCAL, "buf"):
            self._THREAD_LOCAL.buf = ThreadBuffer()

        gene_batch, aln_batch, features, genes = None, None, [], []
        contig_id, contig_seq = contig

        if alns := list(self.target_db.aligner.map(str(contig_seq), buf=self._THREAD_LOCAL.buf)):
            aln_batch = AlignmentBatch.from_mappy(contig_id, len(contig_seq), alns).swap_sides().cull_overlaps()

        if genes := self._gene_finder.find_genes(bytes(contig_seq)):
            num_genes = len(genes)
            starts = np.empty(num_genes, dtype=np.int32)
            ends = np.empty(num_genes, dtype=np.int32)
            strands = np.empty(num_genes, dtype=np.int8)

            for i, g in enumerate(genes):
                starts[i] = g.start
                ends[i] = g.end
                strands[i] = g.strand
                features.append(GenomicFeature(
                    id=gene_id(contig_id, g),
                    type=FeatureType.CDS,
                    segments=[LocationSegment(contig_id, g.start, g.end, Strand(g.strand))]
                ))

            gene_batch = IntervalBatch(starts=starts, ends=ends, strands=strands,
                                       original_indices=np.arange(num_genes, dtype=np.int32)).sort()

        return contig_id, gene_batch, aln_batch, features, genes

    def __call__(self, genome: 'GenomeAssembly', out: Optional[Any] = None) -> Iterable['Locus']:
        """Runs the full pipeline on a genome assembly."""
        alignments, gene_intervals, gene_features, gene_cds = {}, {}, {}, {}

        for contig_id, g_batch, a_batch, features, genes in self._executor.map(self._process_contig, genome):
            if g_batch:
                gene_intervals[contig_id] = g_batch
                gene_features[contig_id] = features
                gene_cds[contig_id] = genes
                if out:
                    out.write_global_genes(contig_id, bytes(genome[contig_id]), genes)
            if a_batch:
                alignments[contig_id] = a_batch

        # for batch in alignments.values():
        #     for n, _ in enumerate(batch.q_names):
        #         print(batch.get_record(n))

        topology_engine = TopologyEngine(
            edges=genome.edges,
            contig_lengths=genome.contig_lengths,
            contig_depths=genome.contig_depths,
            features=gene_intervals,
            mode=self.mode,
            circular=genome.circular)

        builder = LocusBuilder(
            topology_engine=topology_engine,
            genome=genome,
            target_feature_type=self.target_db.feature_type,
            max_feature_hops=self.max_feature_hops,
            locus_tolerance=self.locus_tolerance,
            features=gene_features,
            genes=gene_cds,
            max_flank_distance=self.max_flank_distance
        )

        loci = builder.assemble(alignments, min_coverage=self.min_coverage, min_identity=self.min_identity)
        if self.mode == 'collapse':
            # A walk out of one locus's hub stops at any other locus's hub, so every locus must be built first
            loci = list(loci)
            assign_copy_numbers(loci, genome)
        yield from loci