"""Dual-batch change-merge precheck on top of one saved analysis version.

Two independent batches of ``add`` / ``replace`` / ``remove`` operations
(side A and side B) are each replayed from the **same base version**, exactly
as :mod:`app.simulate` does for a single batch.  The precheck decides whether
the two batches can be merged by applying **A first, then B**, re-running the
longest-prefix-match analysis after every replay.  Nothing is persisted: the
analyzer runs in memory only and no analysis version is written.

Two classes of incompatibility are distinguished:

1. **Structural conflict** — the same ``rule_id`` is operated on by *both*
   batches.  The two batches disagree on the structural fate of one rule (one
   batch appends/rewrites/deletes what the other batch also touches), so there
   is no single A-then-B outcome for that rule.  Such rule ids are reported up
   front with the two operation kinds naming them; the merge itself is not
   attempted (``merged`` is ``null``).

2. **Effective-action conflict** — even when no rule is touched twice, the two
   batches can still disagree on what the final forwarding action of some
   address range ought to be:

   * *Divergence*: a range changed relative to the base on **both** sides ends
     up with a different action on side A than on side B.  The two batches
     demand incompatible actions for the same addresses.  The conflicting
     surface is the exact intersection of the two sides' base-relative changed
     surfaces (IPv4 and IPv6 overlaid independently, never mixed).

   * *Preservation violation*: after applying A then B and recomputing, some
     range that one side changed relative to the base does **not** keep that
     side's action in the merged result — the other batch silently overrode
     the change there (for example B removes a rule that A's new rule relied
     on as a more-specific match).

Conflicting ranges are returned as **minimal CIDR** blocks with the base
action (``null`` for ``none`` = no matching rule) and, where relevant, the
actions seen on the two sides.  All geometry is CIDR-based — addresses are
never enumerated — so IPv6 ranges (up to 2**128) are handled exactly.

When neither conflict class occurs, the response carries the merged analysis
plus its effective-action semantic diff against the base version (same
contract as :func:`app.diff.diff_analyses`).
"""

from __future__ import annotations

import ipaddress
from collections import defaultdict
from typing import Any

from .analyzer import _sort_networks, analyze_rules
from .diff import _effective_partition, _overlay, diff_analyses
from .simulate import PatchError, apply_patch, rebuild_input_rules, validate_patch

Network = ipaddress._BaseNetwork

#: Canonical ordering of the three effective actions (``none`` last).
_ACTION_ORDER: dict[str | None, int] = {"permit": 0, "deny": 1, None: 2}

#: One cell of a base-relative changed surface:
#: ``(network, base_action, side_action)`` — either action may be ``None``
#: (``none``): the base may forward nothing where a rule was added, and a
#: removed rule leaves addresses matching no rule on that side.
ChangedCell = tuple[Network, str | None, str | None]


def _changed_surface(
    base_result: dict[str, Any], side_result: dict[str, Any]
) -> list[ChangedCell]:
    """Cells whose effective action changed between ``base`` and one side.

    The two versions' effective partitions are overlaid geometrically (see
    :func:`app.diff._overlay`); every cell whose base/side actions differs is
    kept, tagged with both actions (``None`` = ``none``, no matching rule —
    including ranges a ``remove`` leaves uncovered).  The cells are pairwise
    disjoint.
    """
    cells: list[ChangedCell] = []
    for version in (4, 6):
        base_pieces = [p for p in _effective_partition(base_result) if p[0].version == version]
        side_pieces = [p for p in _effective_partition(side_result) if p[0].version == version]
        for net, base_action, side_action in _overlay(base_pieces, side_pieces):
            if base_action != side_action:
                cells.append((net, base_action, side_action))
    cells.sort(key=lambda c: (c[0].version, int(c[0].network_address), c[0].prefixlen))
    return cells


def _intersect_changed_surfaces(
    a_cells: list[ChangedCell], b_cells: list[ChangedCell]
) -> list[tuple[Network, str | None, str | None, str | None]]:
    """Exact intersection of the two sides' changed surfaces.

    Returns disjoint cells ``(network, base_action, action_a, action_b)`` for
    addresses changed on **both** sides.  Each changed surface is already a
    pairwise-disjoint CIDR partition, so two of its CIDRs are either equal, in
    a strict containment relation (the intersection is the smaller CIDR), or
    disjoint; IPv4 and IPv6 never intersect.  The two surfaces share the same
    base, so their base actions on any intersection cell necessarily agree.
    Addresses are never enumerated.
    """
    out: list[tuple[Network, str | None, str | None, str | None]] = []
    for a_net, a_base, a_action in a_cells:
        for b_net, b_base, b_action in b_cells:
            if a_net.version != b_net.version:
                continue
            if a_net == b_net:
                out.append((a_net, a_base, a_action, b_action))
            elif b_net.subnet_of(a_net):
                # The stricter B cell's base tag applies on the intersection;
                # it necessarily equals A's enclosing base action.
                out.append((b_net, b_base, a_action, b_action))
            elif a_net.subnet_of(b_net):
                out.append((a_net, a_base, a_action, b_action))
    out.sort(key=lambda c: (c[0].version, int(c[0].network_address), c[0].prefixlen))
    return out


def _surface_vs_partition(
    changed: list[ChangedCell], partition: list[tuple[Network, str | None]]
) -> list[tuple[Network, str | None, str | None, str | None]]:
    """Intersect one side's changed surface with any disjoint partition.

    Returns disjoint cells
    ``(network, base_action, side_action, other_action)`` covering exactly the
    changed surface: the other action is the action of the unique
    equal/enclosing partition CIDR, or ``None`` (``none``) when no partition
    CIDR covers the cell.
    """
    out: list[tuple[Network, str | None, str | None, str | None]] = []
    for net, base_action, side_action in changed:
        match = next((action for p_net, action in partition if net == p_net), None)
        if match is not None:
            out.append((net, base_action, side_action, match))
            continue
        enclosing = next(
            (action for p_net, action in partition if net.subnet_of(p_net)),
            None,
        )
        out.append((net, base_action, side_action, enclosing))
    return out


def _collapse_group(networks: list[Network]) -> list[Network]:
    """Minimum-CIDR representation: merge adjacent equal-prefix siblings."""
    return list(ipaddress.collapse_addresses(_sort_networks(networks)))


def _divergence_conflicts(
    intersection: list[tuple[Network, str | None, str | None, str | None]]
) -> list[dict[str, Any]]:
    """Group the A∩B changed intersection cells whose side actions differ.

    Cells are grouped by ``(family, base_action, action_a, action_b)`` and
    collapsed into minimal CIDRs.
    """
    by_key: dict[tuple[int, str | None, str | None, str | None], list[Network]] = defaultdict(list)
    for net, base_action, a_action, b_action in intersection:
        if a_action != b_action:
            by_key[(net.version, base_action, a_action, b_action)].append(net)

    conflicts: list[dict[str, Any]] = []
    for version, base_action, a_action, b_action in sorted(
        by_key,
        key=lambda k: (k[0], _ACTION_ORDER[k[1]], _ACTION_ORDER[k[2]], _ACTION_ORDER[k[3]]),
    ):
        networks = _collapse_group(by_key[(version, base_action, a_action, b_action)])
        conflicts.append(
            {
                "ip_version": version,
                "base_action": base_action,
                "side_a_action": a_action,
                "side_b_action": b_action,
                "address_count": sum(n.num_addresses for n in networks),
                "range_count": len(networks),
                "cidrs": [str(n) for n in networks],
            }
        )
    return conflicts


def _preservation_conflicts(
    side: str,
    changed: list[ChangedCell],
    merged_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Changed ranges of one side that the merged result fails to preserve.

    The side's changed surface is laid over the merged effective partition;
    every cell whose merged action differs from the side's action is a
    violation, grouped by ``(family, base_action, side_action, merged_action)``
    into minimal CIDRs.
    """
    merged_partition = _effective_partition(merged_result)
    by_key: dict[tuple[int, str | None, str | None, str | None], list[Network]] = defaultdict(list)
    for net, base_action, side_action, merged_action in _surface_vs_partition(
        changed, merged_partition
    ):
        if merged_action != side_action:
            by_key[(net.version, base_action, side_action, merged_action)].append(net)

    conflicts: list[dict[str, Any]] = []
    for version, base_action, side_action, merged_action in sorted(
        by_key,
        key=lambda k: (k[0], _ACTION_ORDER[k[1]], _ACTION_ORDER[k[2]], _ACTION_ORDER[k[3]]),
    ):
        networks = _collapse_group(by_key[(version, base_action, side_action, merged_action)])
        conflicts.append(
            {
                "side": side,
                "ip_version": version,
                "base_action": base_action,
                "expected_action": side_action,
                "merged_action": merged_action,
                "address_count": sum(n.num_addresses for n in networks),
                "range_count": len(networks),
                "cidrs": [str(n) for n in networks],
            }
        )
    return conflicts


def _side_block(ops: dict[str, list[dict[str, Any]]], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ruleset_hash": result.get("ruleset_hash"),
        "analyzer_version": result.get("analyzer_version"),
        "operations": {op: len(items) for op, items in ops.items()},
    }


def _validate_side(
    side: str,
    base_input: list[dict[str, Any]],
    ops: dict[str, list[dict[str, Any]]],
) -> None:
    """Validate one batch as the simulate endpoint would, tagging every issue.

    The caller collects both batches' problems and raises them as one
    :class:`PatchError` so callers can correct A and B simultaneously.
    """
    try:
        validate_patch(base_input, ops["add"], ops["replace"], ops["remove"])
    except PatchError as exc:
        raise PatchError([f"batch {side.upper()}: {issue}" for issue in exc.issues]) from exc


def merge_precheck(
    base_result: dict[str, Any],
    ops_a: dict[str, list[dict[str, Any]]],
    ops_b: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Replay two operation batches from one base and precheck their merge.

    ``ops_a`` / ``ops_b`` each map ``"add"`` / ``"replace"`` / ``"remove"``
    to already schema-validated operation dicts.  Every replay goes through
    :func:`app.simulate.apply_patch` (the existing batch contract) and
    :func:`app.analyzer.analyze_rules` (the existing rule validation), so an
    invalid batch raises :class:`app.simulate.PatchError` and an invalid rule
    raises :class:`app.analyzer.ValidationError`, exactly as the single-batch
    simulate endpoint does.

    Returns a JSON-serializable report (module docstring for the contract);
    nothing about the base version or any other version is persisted.
    """
    base_input = rebuild_input_rules(base_result)

    # ---- Validate both batches against the base before any analysis -------
    # Every offending target of either batch is reported at once, prefixed
    # with its side label.
    issues: list[str] = []
    for side, ops in (("a", ops_a), ("b", ops_b)):
        try:
            _validate_side(side, base_input, ops)
        except PatchError as exc:
            issues.extend(exc.issues)
    if issues:
        raise PatchError(issues)

    # ---- Replay each side independently from the same base ---------------
    patched_a = apply_patch(base_input, **ops_a)
    result_a = analyze_rules(patched_a)
    patched_b = apply_patch(base_input, **ops_b)
    result_b = analyze_rules(patched_b)

    # ---- Structural conflicts: one rule_id touched by both batches --------
    targets = {
        side: {item["rule_id"] for items in ops.values() for item in items}
        for side, ops in (("a", ops_a), ("b", ops_b))
    }
    op_kind_of = {
        side: {item["rule_id"]: op for op, items in ops.items() for item in items}
        for side, ops in (("a", ops_a), ("b", ops_b))
    }
    structural_rule_ids = sorted(targets["a"] & targets["b"])
    structural_conflicts = [
        {
            "rule_id": rule_id,
            "side_a_operation": op_kind_of["a"][rule_id],
            "side_b_operation": op_kind_of["b"][rule_id],
        }
        for rule_id in structural_rule_ids
    ]

    # ---- Each side's base-relative changed surface (IPv4 + IPv6) ---------
    changed_a = _changed_surface(base_result, result_a)
    changed_b = _changed_surface(base_result, result_b)
    intersection = _intersect_changed_surfaces(changed_a, changed_b)
    divergence_conflicts = _divergence_conflicts(intersection)

    base_block = {
        "id": base_result.get("id"),
        "ruleset_hash": base_result.get("ruleset_hash"),
        "analyzer_version": base_result.get("analyzer_version"),
    }
    sides_block = {
        "a": _side_block(ops_a, result_a),
        "b": _side_block(ops_b, result_b),
    }

    # A structural clash means "A then B" has no defined target for that
    # rule; the merge is not attempted.
    if structural_conflicts:
        return {
            "base": base_block,
            "sides": sides_block,
            "compatible": False,
            "status": "structural_conflict",
            "structural_conflicts": structural_conflicts,
            "conflicts": {
                "divergence": divergence_conflicts,
                "merge_preservation": [],
            },
            "merged": None,
            "diff": None,
        }

    # ---- Apply A then B and recompute -------------------------------------
    # Each batch is non-empty on its own and their targets do not overlap, but
    # together they can still remove every base rule (A removes one half, B
    # the other); that leaves no rule to analyze and is rejected as a batch.
    try:
        merged_input = apply_patch(patched_a, **ops_b)
    except PatchError as exc:
        raise PatchError([f"merged result: {issue}" for issue in exc.issues]) from exc
    merged_result = analyze_rules(merged_input)

    # ---- The merged result must preserve each side's changed surface ------
    preservation_conflicts = (
        _preservation_conflicts("a", changed_a, merged_result)
        + _preservation_conflicts("b", changed_b, merged_result)
    )
    preservation_conflicts.sort(
        key=lambda c: (
            c["side"],
            c["ip_version"],
            _ACTION_ORDER[c["base_action"]],
            _ACTION_ORDER[c["expected_action"]],
            _ACTION_ORDER[c["merged_action"]],
        )
    )

    compatible = not divergence_conflicts and not preservation_conflicts
    merged_view = dict(merged_result)
    diff = diff_analyses(base_result, merged_view)
    if compatible:
        return {
            "base": base_block,
            "sides": sides_block,
            "compatible": True,
            "status": "compatible",
            "structural_conflicts": [],
            "conflicts": {
                "divergence": [],
                "merge_preservation": [],
            },
            "merged": merged_view,
            "diff": diff,
        }

    return {
        "base": base_block,
        "sides": sides_block,
        "compatible": False,
        "status": "action_conflict",
        "structural_conflicts": [],
        "conflicts": {
            "divergence": divergence_conflicts,
            "merge_preservation": preservation_conflicts,
        },
        "merged": merged_view,
        "diff": diff,
    }
