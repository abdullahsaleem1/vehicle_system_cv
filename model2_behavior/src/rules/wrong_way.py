"""Rule 2: wrong-way movement against a lane direction.

A track fires when its normalized displacement vector opposes the direction of
the lane it is currently inside by more than ``angle_tolerance_deg`` for at least
``min_consecutive_s`` seconds, and the movement is significant enough
(``min_displacement_bl``) to exclude jitter.

Evaluation is PER LANE: the vehicle is compared against the direction of the lane
whose polygon contains its ground-contact point, and the event's ``zone_id`` is
that lane's name. A vehicle outside every lane cannot be judged wrong-way and is
skipped for that frame.

A lane direction is MANDATORY. Without at least one directional lane the rule
cannot distinguish a wrong-way vehicle from one travelling correctly, so it
REFUSES TO CONSTRUCT.

THRESHOLD STATUS: ALL thresholds are PLACEHOLDER, unvalidated until the Step 8
evaluation harness reports per-rule precision/recall.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from model2_behavior.src.rules.base import FrameContext, Rule, RuleHit
from model2_behavior.src.tracks import TrackState


class WrongWayRule(Rule):
    """Detects vehicles travelling opposite to the configured lane direction."""

    name = "wrong_way"
    required_geometry = ("lane_direction",)

    def __init__(self, config, fps: float) -> None:
        """Initialise and read rule thresholds.

        Args:
            config: Resolved rule configuration.
            fps: Measured frames per second.
        """
        super().__init__(config, fps)
        params = config.params
        self.angle_tolerance_deg = float(params.get("angle_tolerance_deg", 45.0))
        self.min_displacement_bl = float(params.get("min_displacement_bl", 0.15))
        self.direction_window_s = float(params.get("direction_window_s", 0.5))
        # track_id -> consecutive opposing frames
        self._streaks: Dict[int, int] = {}

    def update(self, context: FrameContext, tracks: List[TrackState]) -> Optional[RuleHit]:
        """Evaluate wrong-way conditions for this frame.

        Args:
            context: Frame context.
            tracks: Active tracks.

        Returns:
            A hit when a track has opposed the lane direction long enough.
        """
        self.check_geometry(context.geometry)
        geometry = context.geometry

        for track in tracks:
            track_id = track.track_id
            displacement = track.normalized_displacement(self.direction_window_s)
            if displacement is None:
                self._streaks[track_id] = 0
                continue

            magnitude = math.hypot(displacement[0], displacement[1])
            if magnitude < self.min_displacement_bl:
                self._streaks[track_id] = 0
                continue

            latest = track.latest
            if latest is None:
                self._streaks[track_id] = 0
                continue

            # Evaluate against the lane the vehicle is currently in. A vehicle
            # outside every lane cannot be judged wrong-way.
            lane = geometry.lane_containing(latest.bottom_center)
            if lane is None:
                self._streaks[track_id] = 0
                continue
            direction = lane.direction_vector()
            if direction is None:
                self._streaks[track_id] = 0
                continue

            unit = (displacement[0] / magnitude, displacement[1] / magnitude)
            cosine = unit[0] * direction[0] + unit[1] * direction[1]
            cosine = max(-1.0, min(1.0, cosine))
            angle_deg = math.degrees(math.acos(cosine))

            # Opposing means the angle between motion and lane direction exceeds
            # 180 - tolerance, i.e. the vehicle is heading the other way.
            opposing = angle_deg >= (180.0 - self.angle_tolerance_deg)

            if opposing:
                self._streaks[track_id] = self._streaks.get(track_id, 0) + 1
            else:
                self._streaks[track_id] = 0

            streak = self._streaks.get(track_id, 0)
            if streak < self.min_consecutive_frames:
                continue
            if self.is_debounced(track_id, context.frame_index):
                continue

            self.arm_debounce(track_id, context.frame_index)
            self._streaks[track_id] = 0

            rule_params = self.base_rule_params()
            rule_params.update(
                {
                    "angle_tolerance_deg": self.angle_tolerance_deg,
                    "min_displacement_bl": self.min_displacement_bl,
                    "direction_window_s": self.direction_window_s,
                    "lane_name": lane.name,
                    "lane_direction_unit": [round(direction[0], 6), round(direction[1], 6)],
                    "observed_angle_deg": round(angle_deg, 2),
                    "observed_displacement_bl": round(magnitude, 4),
                    "streak_frames": streak,
                    "threshold_status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
                }
            )
            self.logger.info(
                "wrong_way | track=%s lane=%s angle=%.2f deg (tolerance %.2f) streak=%s",
                track_id,
                lane.name,
                angle_deg,
                self.angle_tolerance_deg,
                streak,
            )
            return RuleHit(
                track_ids=[track_id],
                zone_id=lane.name,
                confidence=float(track.mean_confidence),
                rule_params=rule_params,
                reason=(
                    f"movement {angle_deg:.1f} deg from lane {lane.name} direction "
                    f"for {streak} frames"
                ),
            )

        return None

    def reset(self) -> None:
        """Clear debounce and streak state."""
        super().reset()
        self._streaks.clear()
