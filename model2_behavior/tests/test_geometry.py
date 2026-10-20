"""Geometry parsing, point-in-polygon, and resolution rescaling.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import pytest

from model2_behavior.src.geometry import (
    CameraGeometry,
    GeometryError,
    Lane,
    Polygon,
    build_camera_geometry,
    parse_lane_direction,
    parse_lanes,
    parse_polygon,
    parse_reference_resolution,
    polygon_intersection_area,
    polygon_self_intersects,
)


class TestPolygon:
    """Polygon containment and scaling."""

    def test_needs_three_points(self) -> None:
        with pytest.raises(GeometryError):
            Polygon([(0.0, 0.0), (1.0, 1.0)])

    def test_contains_interior_point(self) -> None:
        square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert square.contains((5.0, 5.0)) is True

    def test_rejects_exterior_point(self) -> None:
        square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert square.contains((15.0, 5.0)) is False

    def test_boundary_point_counts_as_inside(self) -> None:
        square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert square.contains((0.0, 5.0)) is True

    def test_concave_polygon_notch_is_outside(self) -> None:
        # An arrow-head shape with a notch cut into the top edge.
        arrow = Polygon([(0.0, 0.0), (10.0, 0.0), (5.0, 10.0), (2.0, 6.0)])
        assert arrow.contains((5.0, 3.0)) is True
        assert arrow.contains((0.5, 9.0)) is False

    def test_scaled_multiplies_coordinates(self) -> None:
        square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        scaled = square.scaled(2.0, 0.5)
        assert scaled.points == [(0.0, 0.0), (20.0, 0.0), (20.0, 5.0), (0.0, 5.0)]


class TestParsePolygon:
    """Config parsing of ROI polygons."""

    def test_none_returns_none(self) -> None:
        assert parse_polygon(None, "rois.driving_lane") is None

    def test_full_frame_sentinel_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="full_frame"):
            parse_polygon("full_frame", "rois.driving_lane")

    def test_too_few_points_rejected(self) -> None:
        with pytest.raises(GeometryError, match=">= 3"):
            parse_polygon([[0, 0], [1, 1]], "rois.driving_lane")

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(GeometryError):
            parse_polygon([[0, 0], [1, "x"], [2, 2]], "rois.driving_lane")

    def test_valid_polygon_parsed(self) -> None:
        polygon = parse_polygon([[0, 0], [4, 0], [4, 3]], "rois.driving_lane")
        assert polygon is not None
        assert polygon.points == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]


class TestParseLaneDirection:
    """Config parsing of lane direction."""

    def test_none_returns_none(self) -> None:
        assert parse_lane_direction(None, "rois.lane_direction") is None

    def test_flat_four_values_accepted(self) -> None:
        parsed = parse_lane_direction([0, 0, 3, 4], "rois.lane_direction")
        assert parsed == ((0.0, 0.0), (3.0, 4.0))

    def test_nested_two_points_accepted(self) -> None:
        parsed = parse_lane_direction([[0, 0], [3, 4]], "rois.lane_direction")
        assert parsed == ((0.0, 0.0), (3.0, 4.0))

    def test_zero_length_rejected(self) -> None:
        with pytest.raises(GeometryError, match="zero length"):
            parse_lane_direction([5, 5, 5, 5], "rois.lane_direction")


class TestParseReferenceResolution:
    """Config parsing of the reference resolution."""

    def test_none_returns_none(self) -> None:
        assert parse_reference_resolution(None, "rois.reference_resolution") is None

    def test_positive_resolution_accepted(self) -> None:
        assert parse_reference_resolution([1920, 1080], "rois.reference_resolution") == (1920, 1080)

    def test_non_positive_rejected(self) -> None:
        with pytest.raises(GeometryError):
            parse_reference_resolution([0, 1080], "rois.reference_resolution")


class TestRescaling:
    """Resolution rescaling of authored geometry."""

    def test_same_resolution_returns_equivalent(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[Lane("lane_a", Polygon([(0.0, 0.0), (640.0, 0.0), (640.0, 480.0)]))],
        )
        rescaled = geometry.scaled_to((640, 480))
        assert rescaled.lanes[0].polygon.points == geometry.lanes[0].polygon.points

    def test_double_width_doubles_x_only(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[Lane("lane_a", Polygon([(100.0, 100.0), (200.0, 100.0), (200.0, 200.0)]))],
        )
        rescaled = geometry.scaled_to((1280, 480))
        assert rescaled.lanes[0].polygon.points == [(200.0, 100.0), (400.0, 100.0), (400.0, 200.0)]

    def test_direction_rescaled(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[
                Lane(
                    "lane_a",
                    Polygon([(0.0, 0.0), (640.0, 0.0), (640.0, 480.0)]),
                    direction=((100.0, 400.0), (540.0, 400.0)),
                )
            ],
        )
        rescaled = geometry.scaled_to((320, 240))
        assert rescaled.lanes[0].direction == ((50.0, 200.0), (270.0, 200.0))

    def test_no_reference_resolution_is_passthrough(self) -> None:
        geometry = CameraGeometry(
            lanes=[Lane("lane_a", Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))]
        )
        rescaled = geometry.scaled_to((1920, 1080))
        assert rescaled.lanes[0].polygon.points == geometry.lanes[0].polygon.points

    def test_invalid_target_rejected(self) -> None:
        geometry = CameraGeometry(reference_resolution=(640, 480))
        with pytest.raises(GeometryError):
            geometry.scaled_to((0, 480))

    def test_scaled_point_membership_follows_scaling(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[
                Lane(
                    "lane_a",
                    Polygon([(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)]),
                )
            ],
        )
        rescaled = geometry.scaled_to((1280, 960))
        # (1000, 900) is only inside once the polygon is scaled up.
        assert rescaled.lanes[0].polygon.contains((1000.0, 900.0)) is True


class TestDirectionVector:
    """Unit direction vector derivation."""

    def test_direction_is_unit_length(self) -> None:
        lane = Lane(
            "lane_a",
            Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
            direction=((0.0, 0.0), (3.0, 4.0)),
        )
        vector = lane.direction_vector()
        assert vector is not None
        assert vector[0] == pytest.approx(0.6)
        assert vector[1] == pytest.approx(0.8)

    def test_unset_direction_returns_none(self) -> None:
        lane = Lane("lane_a", Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))
        assert lane.direction_vector() is None


class TestLanes:
    """Multiple lanes, each with its own polygon and direction."""

    def test_lane_containing_returns_matching_lane(self) -> None:
        left = Lane("left", Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]))
        right = Lane(
            "right", Polygon([(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)])
        )
        geometry = CameraGeometry(lanes=[left, right])
        assert geometry.lane_containing((50.0, 50.0)) is left
        assert geometry.lane_containing((250.0, 50.0)) is right
        assert geometry.lane_containing((150.0, 50.0)) is None

    def test_lanes_with_direction_filters_unset(self) -> None:
        directed = Lane(
            "a",
            Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
            direction=((0.0, 0.0), (1.0, 0.0)),
        )
        undirected = Lane("b", Polygon([(20.0, 0.0), (30.0, 0.0), (30.0, 10.0)]))
        geometry = CameraGeometry(lanes=[directed, undirected])
        assert geometry.lanes_with_direction() == [directed]

    def test_lane_named(self) -> None:
        lane = Lane("north", Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))
        geometry = CameraGeometry(lanes=[lane])
        assert geometry.lane_named("north") is lane
        assert geometry.lane_named("south") is None


class TestBuildCameraGeometry:
    """Assembly from the config ``rois`` mapping."""

    def test_empty_rois_builds_empty_geometry(self) -> None:
        geometry = build_camera_geometry({})
        assert geometry.has_lanes() is False
        assert geometry.lanes_with_direction() == []

    def test_legacy_keys_are_rejected(self) -> None:
        with pytest.raises(GeometryError, match="no longer supported"):
            build_camera_geometry({"driving_lane": [[0, 0], [10, 0], [10, 10]]})
        with pytest.raises(GeometryError, match="no longer supported"):
            build_camera_geometry({"lane_direction": [0, 0, 1, 1]})

    def test_lanes_parsed_with_directions(self) -> None:
        geometry = build_camera_geometry(
            {
                "lanes": [
                    {
                        "name": "left",
                        "polygon": [[0, 0], [10, 0], [10, 10]],
                        "direction": [0, 0, 1, 0],
                    },
                    {"name": "right", "polygon": [[20, 0], [30, 0], [30, 10]], "direction": None},
                ]
            }
        )
        assert len(geometry.lanes) == 2
        assert geometry.lane_named("left").direction_vector() is not None
        assert geometry.lane_named("right").direction_vector() is None

    def test_zones_parsed(self) -> None:
        geometry = build_camera_geometry({"zones": {"junction": [[0, 0], [10, 0], [10, 10]]}})
        assert "junction" in geometry.zones
        assert geometry.zones["junction"].contains((5.0, 5.0)) is True

    def test_describe_is_serialisable(self) -> None:
        geometry = build_camera_geometry(
            {
                "lanes": [
                    {
                        "name": "left",
                        "polygon": [[0, 0], [10, 0], [10, 10]],
                        "direction": [0, 0, 1, 1],
                    }
                ],
                "reference_resolution": [640, 480],
            }
        )
        described = geometry.describe()
        assert described["reference_resolution"] == [640, 480]
        assert described["lanes"][0]["name"] == "left"
        assert described["lanes"][0]["direction"] == [[0.0, 0.0], [1.0, 1.0]]


class TestParseLanes:
    """Parsing of the multi-lane list."""

    def test_none_returns_empty(self) -> None:
        assert parse_lanes(None, "rois.lanes") == []

    def test_non_list_rejected(self) -> None:
        with pytest.raises(GeometryError, match="list of lanes"):
            parse_lanes("nope", "rois.lanes")

    def test_name_required(self) -> None:
        with pytest.raises(GeometryError, match="name"):
            parse_lanes([{"polygon": [[0, 0], [1, 0], [1, 1]]}], "rois.lanes")

    def test_polygon_required(self) -> None:
        with pytest.raises(GeometryError, match="polygon"):
            parse_lanes([{"name": "a"}], "rois.lanes")

    def test_duplicate_names_rejected(self) -> None:
        entry = {"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}
        with pytest.raises(GeometryError, match="duplicate lane name"):
            parse_lanes([entry, dict(entry)], "rois.lanes")


class TestSelfIntersection:
    """Self-intersection detection used by load-time validation."""

    def test_square_is_not_self_intersecting(self) -> None:
        assert polygon_self_intersects([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]) is False

    def test_bowtie_is_self_intersecting(self) -> None:
        # Vertices in an order that crosses the middle.
        assert polygon_self_intersects([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]) is True

    def test_triangle_cannot_self_intersect(self) -> None:
        assert polygon_self_intersects([(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]) is False


class TestIntersectionArea:
    """Intersection-area computation used for overlap detection."""

    def test_disjoint_polygons_have_zero_area(self) -> None:
        left = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        right = [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)]
        assert polygon_intersection_area(left, right) == pytest.approx(0.0)

    def test_overlapping_rectangles_area(self) -> None:
        left = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        right = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        assert polygon_intersection_area(left, right) == pytest.approx(25.0)

    def test_shared_edge_is_zero_area(self) -> None:
        left = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        right = [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)]
        assert polygon_intersection_area(left, right) == pytest.approx(0.0)


class TestGeometryValidation:
    """Load-time validation of lanes, polygons, and directions."""

    def test_valid_geometry_has_no_problems(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[
                Lane(
                    "left",
                    Polygon([(40.0, 240.0), (300.0, 240.0), (300.0, 470.0), (40.0, 470.0)]),
                    direction=((60.0, 400.0), (280.0, 400.0)),
                ),
                Lane(
                    "right",
                    Polygon([(320.0, 240.0), (600.0, 240.0), (600.0, 470.0), (320.0, 470.0)]),
                    direction=((580.0, 400.0), (340.0, 400.0)),
                ),
            ],
        )
        assert geometry.validate() == []

    def test_self_intersecting_polygon_is_reported(self) -> None:
        geometry = CameraGeometry(
            lanes=[Lane("bowtie", Polygon([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]))]
        )
        assert any("self-intersecting" in problem for problem in geometry.validate())

    def test_polygon_outside_reference_frame_is_reported(self) -> None:
        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[Lane("offframe", Polygon([(100.0, 100.0), (700.0, 100.0), (700.0, 200.0)]))],
        )
        assert any("outside the reference frame" in problem for problem in geometry.validate())

    def test_degenerate_polygon_is_reported(self) -> None:
        geometry = CameraGeometry(
            lanes=[Lane("flat", Polygon([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]))]
        )
        assert any("degenerate" in problem for problem in geometry.validate())

    def test_zero_length_direction_is_reported(self) -> None:
        # Bypass the parser to inject a zero-length direction directly.
        lane = Lane("lane_a", Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))
        lane.direction = ((5.0, 5.0), (5.0, 5.0))
        geometry = CameraGeometry(lanes=[lane])
        assert any("zero length" in problem for problem in geometry.validate())

    def test_duplicate_directions_are_reported(self) -> None:
        geometry = CameraGeometry(
            lanes=[
                Lane(
                    "a",
                    Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
                    direction=((0.0, 0.0), (10.0, 0.0)),
                ),
                Lane(
                    "b",
                    Polygon([(20.0, 0.0), (30.0, 0.0), (30.0, 10.0)]),
                    direction=((20.0, 0.0), (30.0, 0.0)),
                ),
            ]
        )
        assert any("share the same direction vector" in problem for problem in geometry.validate())

    def test_overlapping_lanes_are_reported(self) -> None:
        geometry = CameraGeometry(
            lanes=[
                Lane("a", Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])),
                Lane("b", Polygon([(50.0, 0.0), (150.0, 0.0), (150.0, 100.0), (50.0, 100.0)])),
            ]
        )
        problems = geometry.overlap_problems()
        assert len(problems) == 1
        assert "'a'" in problems[0] and "'b'" in problems[0]

    def test_touching_lanes_are_not_reported_as_overlapping(self) -> None:
        geometry = CameraGeometry(
            lanes=[
                Lane("a", Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])),
                Lane("b", Polygon([(100.0, 0.0), (200.0, 0.0), (200.0, 100.0), (100.0, 100.0)])),
            ]
        )
        assert geometry.overlap_problems() == []

    def test_overlap_tie_break_is_first_lane_in_config_order(self) -> None:
        """The documented deterministic tie-break must be observable."""
        geometry = CameraGeometry(
            lanes=[
                Lane("first", Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])),
                Lane("second", Polygon([(50.0, 0.0), (150.0, 0.0), (150.0, 100.0), (50.0, 100.0)])),
            ]
        )
        # (75, 50) is inside both; config order must win, deterministically.
        assert geometry.lane_containing((75.0, 50.0)).name == "first"
        assert geometry.lane_containing((75.0, 50.0)).name == "first"  # stable on repeat

    def test_overlap_note_is_none_when_no_overlap(self) -> None:
        geometry = CameraGeometry(
            lanes=[Lane("a", Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]))]
        )
        assert geometry.overlap_note() is None


class TestBuildValidation:
    """``build_camera_geometry`` must reject invalid geometry at load time."""

    def test_valid_lanes_pass(self) -> None:
        geometry = build_camera_geometry(
            {
                "lanes": [
                    {
                        "name": "lane_1",
                        "polygon": [[40, 240], [600, 240], [600, 470], [40, 470]],
                        "direction": [100, 400, 540, 400],
                    }
                ],
                "reference_resolution": [640, 480],
            }
        )
        assert geometry.has_lanes() is True

    def test_self_intersecting_polygon_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="self-intersecting"):
            build_camera_geometry(
                {
                    "lanes": [
                        {
                            "name": "bowtie",
                            "polygon": [[0, 0], [100, 100], [100, 0], [0, 100]],
                        }
                    ]
                }
            )

    def test_polygon_outside_reference_frame_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="outside the reference frame"):
            build_camera_geometry(
                {
                    "lanes": [
                        {"name": "off", "polygon": [[0, 0], [700, 0], [700, 100], [0, 100]]}
                    ],
                    "reference_resolution": [640, 480],
                }
            )

    def test_degenerate_polygon_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="degenerate"):
            build_camera_geometry(
                {"lanes": [{"name": "flat", "polygon": [[0, 0], [10, 0], [20, 0]]}]}
            )

    def test_zero_length_direction_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="zero length"):
            build_camera_geometry(
                {
                    "lanes": [
                        {
                            "name": "lane_a",
                            "polygon": [[0, 0], [10, 0], [10, 10]],
                            "direction": [5, 5, 5, 5],
                        }
                    ]
                }
            )

    def test_overlapping_lanes_are_warned_but_allowed(self) -> None:
        # Overlaps are resolvable (first lane in config order wins), so the
        # loader warns rather than failing; the authoring tool rejects them.
        geometry = build_camera_geometry(
            {
                "lanes": [
                    {"name": "a", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                    {"name": "b", "polygon": [[50, 0], [150, 0], [150, 100], [50, 100]]},
                ]
            }
        )
        assert geometry.overlap_problems() != []
        assert geometry.lane_containing((75.0, 50.0)).name == "a"

    def test_validation_can_be_disabled_for_inspection(self) -> None:
        geometry = build_camera_geometry(
            {"lanes": [{"name": "flat", "polygon": [[0, 0], [10, 0], [20, 0]]}]},
            validate=False,
        )
        assert geometry.validate() != []
