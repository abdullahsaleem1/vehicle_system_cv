"""Rule 1: stopped vehicle inside a driving-lane ROI.

A track fires when its normalized speed stays below ``speed_thresh_bl_s`` for at
least ``min_consecutive_s`` seconds AND its ground-contact point lies inside ONE
of the configured lanes under ``rois.lanes``. The event's ``zone_id`` is the name
of the lane that matched; overlapping lanes resolve to the first match in config
order.

Lane geometry is MANDATORY: ``full_frame`` is rejected. A vehicle stopped outside
a designated lane is not an incident, so the rule refuses to construct without
geometry.

THRESHOLD STATUS: ``speed_thresh_bl_s`` is a PLACEHOLDER, unvalidated until the
Step 8 evaluation harness reports per-rule precision/recall.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from model2_behavior.src.rules.base import FrameContext, Rule, RuleHit
from model2_behavior.src.tracks import TrackState


class StoppedVehicleRule(Rule):
    """Detects vehicles stationary inside a driving lane."""

    name = "stopped_vehicle"
    required_geometry = ("driving_lane",)

    def __init__(self, config, fps: float) -> None:
        """Initialise and read rule thresholds.

        Args:
            config: Resolved rule configuration.
            fps: Measured frames per second.
        """
        super().__init__(config, fps)
        params = config.params
        self.speed_thresh_bl_s = float(params.get("speed_thresh_bl_s", 0.05))
        self.speed_window_s = float(params.get("speed_window_s", 0.5))
        self.require_roi = bool(params.get("require_roi", True))
        # track_id -> consecutive qualifying frames
        self._streaks: Dict[int, int] = {}

    def update(self, context: FrameContext, tracks: List[TrackState]) -> Optional[RuleHit]:
        """Evaluate stopped-vehicle conditions for this frame.

        Args:
            context: Frame context.
            tracks: Active tracks.

        Returns:
            A hit when a track has been stationary in-lane long enough.
        """
        self.check_geometry(context.geometry)
        geometry = context.geometry

        for track in tracks:
            track_id = track.track_id
            speed = track.speed_bl_s(self.speed_window_s)
            latest = track.latest
            if speed is None or latest is None:
                self._streaks[track_id] = 0
                continue

            # Which lane, if any, contains this vehicle's ground-contact point?
            matched_lane = None
            if self.require_roi:
                matched_lane = geometry.lane_containing(latest.bottom_center)

            if speed <= self.speed_thresh_bl_s and (matched_lane is not None or not self.require_roi):
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

            zone_id = matched_lane.name if matched_lane is not None else None
            rule_params = self.base_rule_params()
            rule_params.update(
                {
                    "speed_thresh_bl_s": self.speed_thresh_bl_s,
                    "speed_window_s": self.speed_window_s,
                    "observed_speed_bl_s": round(speed, 4),
                    "require_roi": self.require_roi,
                    "lane": matched_lane.as_dict() if matched_lane is not None else None,
                    "streak_frames": streak,
                    "threshold_status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
                }
            )
            if context.geometry_describe.get("reference_resolution"):
                rule_params["reference_resolution"] = context.geometry_describe["reference_resolution"]

            self.logger.info(
                "stopped_vehicle | track=%s lane=%s speed=%.4f bl/s (thresh %.4f) streak=%s",
                track_id,
                zone_id,
                speed,
                self.speed_thresh_bl_s,
                streak,
            )
            return RuleHit(
                track_ids=[track_id],
                zone_id=zone_id,
                confidence=float(track.mean_confidence),
                rule_params=rule_params,
                reason=(
                    f"speed {speed:.4f} bl/s <= {self.speed_thresh_bl_s} for {streak} frames "
                    f"in lane {zone_id}"
                ),
            )

        return None

    def reset(self) -> None:
        """Clear debounce and streak state."""
        super().reset()
        self._streaks.clear()
