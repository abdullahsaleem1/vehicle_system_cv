"""Camera geometry: multiple lanes, lane directions, and resolution rescaling.

A camera may have any number of lanes. Each lane carries its own polygon and its
own direction vector; ``rois.lanes[].direction`` is the SINGLE SOURCE OF TRUTH for
lane direction, so a direction can never drift out of sync with its lane.

Geometry is authored in the PIXEL coordinates of a reference frame resolution
(see ``scripts/define_camera_geometry.py``). At runtime the live stream may have
a different resolution, so every geometry value is rescaled from the stored
reference resolution to the actual frame size.

Coordinate convention: ``(0, 0)`` is the top-left of the frame, ``x`` increases
right, ``y`` increases down (image coordinates).

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model2_behavior.src.logging_setup import get_logger

logger = get_logger(__name__)

Point = Tuple[float, float]

# Sentinel meaning "no geometry configured". Rejected for required geometry.
FULL_FRAME = "full_frame"

#: A shared polygon edge shorter than this (px) is treated as "touching", not
#: "overlapping", so neighbouring lanes that share a boundary line do not get
#: flagged. Intersection area must exceed this to count as real overlap.
OVERLAP_AREA_TOLERANCE = 1e-6
#: Point-in-polygon test offsets used to classify a shared boundary.
_EDGE_OFFSET = 1e-4


class GeometryError(ValueError):
    """Raised when required camera geometry is missing or malformed."""


# ---------------------------------------------------------------------------
# Low-level geometric predicates (used by Polygon and by load-time validation).
# ---------------------------------------------------------------------------


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Return the signed area of triangle ``a-b-c`` (cross product z).

    Args:
        a: First point.
        b: Second point.
        c: Third point.

    Returns:
        Positive for a counter-clockwise turn, negative for clockwise, and 0
        when the points are collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Return True if segments ``a1-a2`` and ``b1-b2`` cross or touch.

    Uses orientation tests, so collinear touching is reported as intersecting.
    Endpoints shared between two adjacent polygon edges are handled by the
    caller skipping adjacent edge pairs.

    Args:
        a1: Segment A start.
        a2: Segment A end.
        b1: Segment B start.
        b2: Segment B end.

    Returns:
        True if the segments intersect or touch.
    """
    d1 = _orientation(b1, b2, a1)
    d2 = _orientation(b1, b2, a2)
    d3 = _orientation(a1, a2, b1)
    d4 = _orientation(a1, a2, b2)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if d1 == 0 and _on_segment(a1, b1, b2):
        return True
    if d2 == 0 and _on_segment(a2, b1, b2):
        return True
    if d3 == 0 and _on_segment(b1, a1, a2):
        return True
    if d4 == 0 and _on_segment(b2, a1, a2):
        return True
    return False


def _signed_area(points: Sequence[Point]) -> float:
    """Return the signed area of a polygon (shoelace formula).

    Args:
        points: Ordered polygon vertices.

    Returns:
        Positive for counter-clockwise winding, negative for clockwise.
    """
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_self_intersects(points: Sequence[Point]) -> bool:
    """Return True if a polygon's edges cross each other.

    Adjacent edges are skipped because they legitimately share a vertex, and the
    closing edge is compared against the first edge as a pair.

    Args:
        points: Ordered polygon vertices.

    Returns:
        True when any non-adjacent edge pair intersects.
    """
    count = len(points)
    if count < 4:
        # A triangle cannot self-intersect. A degenerate 3-point "polygon" with
        # collinear points is caught by the area check instead.
        return False

    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            # Skip edges that share a vertex.
            if first == second or first_next == second or second_next == first:
                continue
            if _segments_intersect(
                points[first], points[first_next], points[second], points[second_next]
            ):
                return True
    return False


def _clip_polygon(
    subject: Sequence[Point], clip: Sequence[Point]
) -> List[Point]:
    """Intersect two convex-or-concave polygons via Sutherland-Hodgman clipping.

    The clip polygon is used as the clipping window, so the result is the
    intersection. This only yields the correct area when ``clip`` is convex;
    for the concave case the caller must not rely on it, which is why overlap
    validation is documented as exact for axis-aligned rectangles.

    Args:
        subject: Polygon to clip.
        clip: Clipping polygon.

    Returns:
        Vertices of the intersection polygon (possibly empty).
    """
    output: List[Point] = list(subject)
    count = len(clip)
    # Orientation of the clip polygon decides the "inside" side of each edge.
    orientation = 1.0 if _signed_area(clip) >= 0 else -1.0

    for index in range(count):
        if not output:
            return []
        edge_start = clip[index]
        edge_end = clip[(index + 1) % count]
        input_list = output
        output = []
        previous = input_list[-1]
        previous_inside = orientation * _orientation(edge_start, edge_end, previous) >= 0
        for current in input_list:
            current_inside = orientation * _orientation(edge_start, edge_end, current) >= 0
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    return output


def _line_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point:
    """Return the intersection point of lines ``p1-p2`` and ``q1-q2``.

    Args:
        p1: First line point.
        p2: Second line point.
        q1: Other line's first point.
        q2: Other line's second point.

    Returns:
        Intersection point; falls back to ``p2`` when the lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    determinant_p = x1 * y2 - y1 * x2
    determinant_q = x3 * y4 - y3 * x4
    return (
        (determinant_p * (x3 - x4) - (x1 - x2) * determinant_q) / denominator,
        (determinant_p * (y3 - y4) - (y1 - y2) * determinant_q) / denominator,
    )


def polygon_intersection_area(points_a: Sequence[Point], points_b: Sequence[Point]) -> float:
    """Return the area of the intersection of two polygons.

    Exact for convex polygons (including axis-aligned rectangles); an
    approximation for concave ones. It is used only to decide whether two lanes
    meaningfully overlap, and concave lane shapes with tiny sliver intersections
    are not a realistic input.

    Args:
        points_a: First polygon's vertices.
        points_b: Second polygon's vertices.

    Returns:
        Intersection area in pixel-squared units, never negative.
    """
    if len(points_a) < 3 or len(points_b) < 3:
        return 0.0
    clipped = _clip_polygon(points_a, points_b)
    if len(clipped) < 3:
        return 0.0
    return abs(_signed_area(clipped))


def _touches_but_does_not_overlap(points_a: Sequence[Point], points_b: Sequence[Point]) -> bool:
    """Return True when two polygons share only a boundary or a vertex.

    A shared wall between neighbouring lanes is a normal layout, so it must not
    be reported as an overlap. Any shared edge or vertex implies an intersection
    area of zero, which is what this function distinguishes from a real sliver.

    Args:
        points_a: First polygon's vertices.
        points_b: Second polygon's vertices.

    Returns:
        True if the polygons touch without enclosing a shared area.
    """
    return polygon_intersection_area(points_a, points_b) <= OVERLAP_AREA_TOLERANCE


def polygon_inside_frame(points: Sequence[Point], width: int, height: int) -> bool:
    """Return True if every vertex lies within a ``width`` x ``height`` frame.

    Args:
        points: Polygon vertices.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        True when all vertices satisfy ``0 <= x <= width`` and ``0 <= y <= height``.
    """
    for x, y in points:
        if x < 0.0 or y < 0.0 or x > float(width) or y > float(height):
            return False
    return True



@dataclass
class Polygon:
    """A closed polygon in pixel coordinates."""

    points: List[Point]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise GeometryError(f"A polygon needs >= 3 points, got {len(self.points)}")

    def contains(self, point: Point) -> bool:
        """Return True if ``point`` lies inside the polygon.

        Uses the even-odd (ray casting) rule, with an on-edge check so points
        exactly on the boundary count as inside.

        Args:
            point: ``(x, y)`` in the same coordinate space as ``self.points``.

        Returns:
            True if inside or on the boundary.
        """
        x, y = point
        inside = False
        count = len(self.points)
        for index in range(count):
            x1, y1 = self.points[index]
            x2, y2 = self.points[(index + 1) % count]
            if _on_segment((x, y), (x1, y1), (x2, y2)):
                return True
            if (y1 > y) != (y2 > y):
                denominator = (y2 - y1) if (y2 - y1) != 0 else 1e-12
                x_intersect = (x2 - x1) * (y - y1) / denominator + x1
                if x < x_intersect:
                    inside = not inside
        return inside

    def centroid(self) -> Point:
        """Return the mean of the polygon vertices."""
        count = len(self.points)
        return (
            sum(p[0] for p in self.points) / count,
            sum(p[1] for p in self.points) / count,
        )

    def scaled(self, sx: float, sy: float) -> "Polygon":
        """Return a copy scaled by ``sx``/``sy``."""
        return Polygon([(x * sx, y * sy) for x, y in self.points])

    def as_list(self) -> List[List[float]]:
        """Return a JSON-serialisable nested list."""
        return [[float(x), float(y)] for x, y in self.points]


def _on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-9) -> bool:
    """Return True if ``point`` lies on segment ``start``-``end``."""
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tolerance * max(1.0, abs(x2 - x1) + abs(y2 - y1)):
        return False
    if min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and min(y1, y2) - tolerance <= y <= max(
        y1, y2
    ) + tolerance:
        return True
    return False


@dataclass
class Lane:
    """One driving lane: a polygon plus the direction traffic should flow in.

    Attributes:
        name: Lane identifier, used as ``zone_id`` in events.
        polygon: The lane outline in pixel coordinates.
        direction: Direction of travel as ``((x1, y1), (x2, y2))`` (tail -> head),
            or None when not authored. This is the SINGLE SOURCE OF TRUTH for
            lane direction; no other config key carries it.
    """

    name: str
    polygon: Polygon
    direction: Optional[Tuple[Point, Point]] = None

    def direction_vector(self) -> Optional[Point]:
        """Return the unit direction vector for this lane, or None if unset.

        Returns:
            ``(dx, dy)`` normalized to unit length, or None when unset or
            degenerate.
        """
        if self.direction is None:
            return None
        (x1, y1), (x2, y2) = self.direction
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return None
        return (dx / norm, dy / norm)

    def scaled(self, sx: float, sy: float) -> "Lane":
        """Return a copy scaled by ``sx``/``sy``.

        Args:
            sx: Horizontal scale factor.
            sy: Vertical scale factor.

        Returns:
            A rescaled lane.
        """
        return Lane(
            name=self.name,
            polygon=self.polygon.scaled(sx, sy),
            direction=(
                (
                    (self.direction[0][0] * sx, self.direction[0][1] * sy),
                    (self.direction[1][0] * sx, self.direction[1][1] * sy),
                )
                if self.direction
                else None
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable form.

        Returns:
            Dict with name, polygon, and direction.
        """
        return {
            "name": self.name,
            "polygon": self.polygon.as_list(),
            "direction": [list(self.direction[0]), list(self.direction[1])] if self.direction else None,
        }


@dataclass
class CameraGeometry:
    """Resolved geometry for one camera, with optional resolution rescaling.

    A camera may have MULTIPLE lanes. Each lane carries its own polygon and
    direction, so ``wrong_way`` can be evaluated per lane.

    Attributes:
        reference_resolution: Resolution the pixel values were authored in.
        lanes: Ordered list of lanes.
        zones: Additional named polygons that are not driving lanes.
    """

    reference_resolution: Optional[Tuple[int, int]] = None
    lanes: List[Lane] = field(default_factory=list)
    zones: Dict[str, Polygon] = field(default_factory=dict)

    def scaled_to(self, frame_size: Tuple[int, int]) -> "CameraGeometry":
        """Return a copy rescaled to ``frame_size`` ``(width, height)``.

        Scaling is applied only when a reference resolution is recorded and it
        differs from the target. This keeps authored pixel values authoritative.

        Args:
            frame_size: Target ``(width, height)`` in pixels.

        Returns:
            A rescaled :class:`CameraGeometry` (or self-equivalent copy).

        Raises:
            GeometryError: If ``frame_size`` is not positive.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise GeometryError(f"Invalid frame size: {frame_size}")

        if self.reference_resolution is None:
            return CameraGeometry(
                reference_resolution=None,
                lanes=[Lane(lane.name, lane.polygon, lane.direction) for lane in self.lanes],
                zones=dict(self.zones),
            )

        ref_w, ref_h = self.reference_resolution
        if ref_w <= 0 or ref_h <= 0:
            raise GeometryError(f"Invalid reference resolution: {self.reference_resolution}")

        sx, sy = width / ref_w, height / ref_h
        if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
            return self

        return CameraGeometry(
            reference_resolution=(width, height),
            lanes=[lane.scaled(sx, sy) for lane in self.lanes],
            zones={name: poly.scaled(sx, sy) for name, poly in self.zones.items()},
        )

    def has_lanes(self) -> bool:
        """Return True if at least one driving lane is configured."""
        return len(self.lanes) > 0

    def lanes_with_direction(self) -> List[Lane]:
        """Return lanes that have a usable direction configured.

        Returns:
            Lanes whose :meth:`Lane.direction_vector` is not None.
        """
        return [lane for lane in self.lanes if lane.direction_vector() is not None]

    def validate(self, where: str = "rois") -> List[str]:
        """Return a list of geometry problems. Empty means valid.

        Called at load time so a bad geometry fails loudly and early rather than
        producing silent misses during a run. Checks performed:

            * every polygon has at least 3 points (enforced by ``Polygon``);
            * polygons are non-degenerate and do not self-intersect;
            * every polygon lies inside the reference frame, when one is known;
            * lane direction vectors are non-zero and distinct from each other;
            * overlapping lanes are reported.

        Args:
            where: Location string used in messages.

        Returns:
            List of human-readable problems.

        Note:
            Overlaps are reported as problems. Overlapping lanes are only
            *resolvable*, not *valid*: the resolver in
            :meth:`lane_containing` breaks ties deterministically (see its
            docstring), but a camera where two lane polygons genuinely cover the
            same ground usually means the authoring was wrong. Authoring tools
            should therefore surface this list and refuse to save.
        """
        problems: List[str] = []

        for index, lane in enumerate(self.lanes):
            lane_where = f"{where}.lanes[{index}] ({lane.name})"
            points = lane.polygon.points

            if len(points) < 3:
                problems.append(f"{lane_where}: polygon needs >= 3 points, got {len(points)}")
                continue

            if abs(_signed_area(points)) <= 1e-9:
                problems.append(
                    f"{lane_where}: polygon is degenerate (zero area); its points are collinear"
                )

            if polygon_self_intersects(points):
                problems.append(f"{lane_where}: polygon is self-intersecting")

            if self.reference_resolution is not None:
                width, height = self.reference_resolution
                if not polygon_inside_frame(points, width, height):
                    problems.append(
                        f"{lane_where}: polygon has a point outside the reference frame "
                        f"{width}x{height}"
                    )

            direction = lane.direction
            if direction is not None:
                (x1, y1), (x2, y2) = direction
                if math.hypot(x2 - x1, y2 - y1) < 1e-9:
                    problems.append(
                        f"{lane_where}: direction vector has zero length (its two points coincide)"
                    )

        # Duplicate directions across lanes are reported separately because they
        # usually mean one lane was copied and its direction never updated.
        seen_directions: Dict[Tuple[float, float], str] = {}
        for lane in self.lanes:
            vector = lane.direction_vector()
            if vector is None:
                continue
            key = (round(vector[0], 6), round(vector[1], 6))
            if key in seen_directions:
                problems.append(
                    f"{where}: lanes {seen_directions[key]!r} and {lane.name!r} share the same "
                    "direction vector; a copied lane's direction was probably not updated"
                )
            else:
                seen_directions[key] = lane.name

        problems.extend(self.overlap_problems(where))
        return problems

    def overlap_problems(self, where: str = "rois") -> List[str]:
        """Return a problem string for every overlapping lane pair.

        Args:
            where: Location string used in messages.

        Returns:
            One message per pair whose polygons enclose a genuinely shared area.
            Shared boundaries and shared vertices (the normal way to lay out
            adjacent lanes) are NOT reported; only a real shared region is.
        """
        problems: List[str] = []
        for first in range(len(self.lanes)):
            for second in range(first + 1, len(self.lanes)):
                lane_a = self.lanes[first]
                lane_b = self.lanes[second]
                area = polygon_intersection_area(lane_a.polygon.points, lane_b.polygon.points)
                if area > OVERLAP_AREA_TOLERANCE:
                    problems.append(
                        f"{where}.lanes: {lane_a.name!r} and {lane_b.name!r} overlap "
                        f"(intersection area {area:.3f} px^2). Tie-break: the FIRST lane in "
                        "config order wins, so a vehicle in the shared region is attributed to "
                        f"{lane_a.name!r}. Non-convex polygon intersections are approximate."
                    )
        return problems

    def overlap_note(self, where: str = "rois") -> Optional[str]:
        """Return a single-line summary of overlapping lanes, or None.

        Used by the runtime to log overlaps without failing the run, because an
        overlap is resolvable (deterministically) rather than fatal.

        Args:
            where: Location string used in messages.

        Returns:
            A summary string when any pair overlaps, otherwise None.
        """
        problems = self.overlap_problems(where)
        if not problems:
            return None
        return " ".join(problems)

    def lane_named(self, name: str) -> Optional[Lane]:
        """Return a lane by name, or None.

        Args:
            name: Lane name.

        Returns:
            The matching lane, or None.
        """
        for lane in self.lanes:
            if lane.name == name:
                return lane
        return None

    def lane_containing(self, point: Point) -> Optional[Lane]:
        """Return the first lane whose polygon contains ``point``, or None.

        TIE-BREAK: when lane polygons overlap, the FIRST lane in config order
        wins. This is deterministic and independent of dict ordering, so a given
        point always resolves to the same lane. Overlaps are flagged by
        :meth:`validate` and can be inspected with :meth:`overlap_note`.

        Args:
            point: ``(x, y)`` in the same coordinate space as the lane polygons.

        Returns:
            The matching lane, or None.
        """
        for lane in self.lanes:
            if lane.polygon.contains(point):
                return lane
        return None

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary for logging and event params.

        Returns:
            Dict with the reference resolution, lanes, and zones.
        """
        return {
            "reference_resolution": list(self.reference_resolution) if self.reference_resolution else None,
            "lanes": [lane.as_dict() for lane in self.lanes],
            "zones": {name: poly.as_list() for name, poly in self.zones.items()},
        }


def parse_polygon(raw: Any, where: str) -> Optional[Polygon]:
    """Parse a polygon from config, rejecting the ``full_frame`` sentinel.

    Args:
        raw: Nested list of ``[x, y]`` pairs, or None, or ``"full_frame"``.
        where: Location string used in error messages.

    Returns:
        A :class:`Polygon`, or None when ``raw`` is None.

    Raises:
        GeometryError: If the value is malformed or uses the sentinel.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == FULL_FRAME:
            raise GeometryError(
                f"{where}: 'full_frame' is not accepted as a driving-lane ROI; "
                "stopped_vehicle requires an explicit lane polygon."
            )
        raise GeometryError(f"{where}: expected a list of [x, y] points, got {raw!r}")

    if not isinstance(raw, (list, tuple)):
        raise GeometryError(f"{where}: expected a list of [x, y] points, got {type(raw).__name__}")

    points: List[Point] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise GeometryError(f"{where}[{index}]: expected [x, y], got {item!r}")
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"{where}[{index}]: non-numeric coordinates {item!r}") from exc

    if len(points) < 3:
        raise GeometryError(f"{where}: polygon needs >= 3 points, got {len(points)}")
    return Polygon(points)


def parse_lane_direction(raw: Any, where: str) -> Optional[Tuple[Point, Point]]:
    """Parse lane direction from config.

    Accepts ``[x1, y1, x2, y2]`` or ``[[x1, y1], [x2, y2]]``.

    Args:
        raw: Raw config value, or None.
        where: Location string used in error messages.

    Returns:
        ``((x1, y1), (x2, y2))`` or None.

    Raises:
        GeometryError: If the value is malformed or degenerate (zero length).
    """
    if raw is None:
        return None

    coords: Sequence[Any]
    if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(
        isinstance(v, (int, float)) for v in raw
    ):
        coords = [(raw[0], raw[1]), (raw[2], raw[3])]
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            coords = [((float(raw[0][0]), float(raw[0][1]))), (float(raw[1][0]), float(raw[1][1]))]
        except (TypeError, ValueError, IndexError) as exc:
            raise GeometryError(f"{where}: malformed lane_direction {raw!r}") from exc
    else:
        raise GeometryError(
            f"{where}: lane_direction must be [x1,y1,x2,y2] or [[x1,y1],[x2,y2]], got {raw!r}"
        )

    start, end = (float(coords[0][0]), float(coords[0][1])), (float(coords[1][0]), float(coords[1][1]))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-9:
        raise GeometryError(f"{where}: lane_direction has zero length (points coincide)")
    return (start, end)


def parse_reference_resolution(raw: Any, where: str) -> Optional[Tuple[int, int]]:
    """Parse a reference resolution ``[width, height]`` from config.

    Args:
        raw: Raw config value, or None.
        where: Location string used in error messages.

    Returns:
        ``(width, height)`` or None.

    Raises:
        GeometryError: If the value is malformed or non-positive.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise GeometryError(f"{where}: expected [width, height], got {raw!r}")
    try:
        width, height = int(raw[0]), int(raw[1])
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{where}: non-integer resolution {raw!r}") from exc
    if width <= 0 or height <= 0:
        raise GeometryError(f"{where}: resolution must be positive, got {raw!r}")
    return (width, height)


def parse_lanes(raw_lanes: Any, where: str) -> List[Lane]:
    """Parse the ``rois.lanes`` list into :class:`Lane` objects.

    Each entry must be a mapping with ``name``, ``polygon``, and an optional
    ``direction``. This list is the SINGLE SOURCE OF TRUTH for both lane
    polygons and lane directions; no other config key carries a direction.

    Args:
        raw_lanes: The ``rois.lanes`` value: a list of lane mappings, or None.
        where: Location string used in error messages.

    Returns:
        Ordered list of lanes.

    Raises:
        GeometryError: If the list or any entry is malformed, or two lanes share
            a name.
    """
    if raw_lanes is None:
        return []
    if not isinstance(raw_lanes, (list, tuple)):
        raise GeometryError(f"{where}: expected a list of lanes, got {type(raw_lanes).__name__}")

    lanes: List[Lane] = []
    seen_names: set[str] = set()

    for index, entry in enumerate(raw_lanes):
        entry_where = f"{where}[{index}]"
        if not isinstance(entry, dict):
            raise GeometryError(f"{entry_where}: expected a mapping with name/polygon/direction")

        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise GeometryError(f"{entry_where}: 'name' is required and must be a non-empty string")
        if name in seen_names:
            raise GeometryError(f"{entry_where}: duplicate lane name {name!r}")
        seen_names.add(name)

        polygon = parse_polygon(entry.get("polygon"), f"{entry_where}.polygon")
        if polygon is None:
            raise GeometryError(f"{entry_where}: 'polygon' is required for each lane")

        direction = parse_lane_direction(entry.get("direction"), f"{entry_where}.direction")
        lanes.append(Lane(name=name, polygon=polygon, direction=direction))

    return lanes


def build_camera_geometry(
    rois: Dict[str, Any], where: str = "rois", validate: bool = True
) -> CameraGeometry:
    """Build a :class:`CameraGeometry` from the config ``rois`` mapping.

    Recognised keys:
        ``lanes``: list of ``{name, polygon, direction}`` mappings. Each lane has
            its own polygon and its own direction vector.
        ``zones``: additional named polygons that are not driving lanes.
        ``reference_resolution``: ``[width, height]`` the pixels were authored in.

    The legacy single-lane keys ``driving_lane`` and ``lane_direction`` are
    REJECTED with an explicit error pointing at the new structure, so a stale
    config cannot silently half-work.

    When ``validate`` is True, the assembled geometry is checked for the problems
    listed in :meth:`CameraGeometry.validate` and any of them raises
    :class:`GeometryError`. Overlapping lanes are the one exception: they are
    resolvable by a documented deterministic tie-break, so they are logged as a
    warning instead of rejected. The authoring tool surfaces them as errors.

    Args:
        rois: The ``rois`` mapping from ``model2.yaml``.
        where: Location string used in error messages.
        validate: Run structural validation on the assembled geometry.

    Returns:
        A populated :class:`CameraGeometry`.

    Raises:
        GeometryError: If any geometry value is malformed, a legacy key is used,
            or (with ``validate``) a lane polygon/direction is invalid.
    """
    for legacy in ("driving_lane", "lane_direction"):
        if legacy in rois:
            raise GeometryError(
                f"{where}.{legacy} is no longer supported; define lanes under "
                f"{where}.lanes as a list of {{name, polygon, direction}} mappings. "
                "Lane direction lives only in the lane entry."
            )

    zones_raw = rois.get("zones") or {}
    if not isinstance(zones_raw, dict):
        raise GeometryError(f"{where}.zones must be a mapping")

    zones: Dict[str, Polygon] = {}
    for name, raw_polygon in zones_raw.items():
        polygon = parse_polygon(raw_polygon, f"{where}.zones.{name}")
        if polygon is not None:
            zones[name] = polygon

    geometry = CameraGeometry(
        reference_resolution=parse_reference_resolution(
            rois.get("reference_resolution"), f"{where}.reference_resolution"
        ),
        lanes=parse_lanes(rois.get("lanes"), f"{where}.lanes"),
        zones=zones,
    )

    if validate:
        overlap_messages = set(geometry.overlap_problems(where))
        problems = [problem for problem in geometry.validate(where) if problem not in overlap_messages]
        if problems:
            raise GeometryError(
                f"Invalid camera geometry at {where}: " + "; ".join(problems)
            )
        note = geometry.overlap_note(where)
        if note:
            logger.warning("Lane geometry warning | %s", note)

    return geometry
