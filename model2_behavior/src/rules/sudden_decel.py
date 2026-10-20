"""Rule 3: sudden deceleration.

A track fires when its normalized speed drops by more than
``decel_thresh_bl_s2`` (body-lengths per second squared) within ``window_s``,
having been genuinely moving (``min_speed_before_bl_s``) beforehand.

Deceleration is computed from the speed history inside the window, not from two
adjacent frames, so single-frame detector jitter does not trigger it.

THRESHOLD STATUS: ALL thresholds are PLACEHOLDER, unvalidated until the Step 8
evaluation harness reports per-rule precision/recall.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from model2_behavior.src.rules.base import FrameContext, Rule, RuleHit
from model2_behavior.src.tracks import TrackState


class SuddenDecelRule(Rule):
    """Detects sharp speed drops within a short window."""

    name = "sudden_decel"
    required_geometry = ()

    def __init__(self, config, fps: float) -> None:
        """Initialise and read rule thresholds.

        Args:
            config: Resolved rule configuration.
            fps: Measured frames per second.
        """
        super().__init__(config, fps)
        params = config.params
        self.decel_thresh_bl_s2 = float(params.get("decel_thresh_bl_s2", 1.5))
        self.window_s = float(params.get("window_s", 1.0))
        self.min_speed_before_bl_s = float(params.get("min_speed_before_bl_s", 0.30))
        # track_id -> consecutive qualifying frames
        self._streaks: Dict[int, int] = {}

    def _estimate_deceleration(self, track: TrackState) -> Optional[tuple[float, float, float]]:
        """Estimate deceleration over the configured window.

        Splits the window in half and compares mean speed of the earlier half to
        the later half, which is more robust than a two-point difference.

        Args:
            track: Track to analyse.

        Returns:
            ``(deceleration_bl_s2, speed_before, speed_after)`` or None when
            there is insufficient history.
        """
        window = track.history_spanning(self.window_s)
        if len(window) < 3:
            return None

        midpoint = len(window) // 2
        earlier = window[:midpoint]
        later = window[midpoint:]
        if len(earlier) < 2 or len(later) < 2:
            return None

        def mean_speed(observations) -> Optional[float]:
            first, last = observations[0], observations[-1]
            elapsed = last.time_s - first.time_s
            if elapsed <= 1e-9:
                return None
            (x1, y1) = first.center
            (x2, y2) = last.center
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            mean_height = sum(obs.height for obs in observations) / len(observations)
            return (distance / elapsed) / mean_height

        speed_before = mean_speed(earlier)
        speed_after = mean_speed(later)
        if speed_before is None or speed_after is None:
            return None

        elapsed = later[-1].time_s - earlier[0].time_s
        if elapsed <= 1e-9:
            return None

        deceleration = (speed_before - speed_after) / elapsed
        return (deceleration, speed_before, speed_after)

    def update(self, context: FrameContext, tracks: List[TrackState]) -> Optional[RuleHit]:
        """Evaluate deceleration conditions for this frame.

        Args:
            context: Frame context.
            tracks: Active tracks.

        Returns:
            A hit when a track decelerates sharply enough.
        """
        for track in tracks:
            track_id = track.track_id
            estimate = self._estimate_deceleration(track)
            if estimate is None:
                self._streaks[track_id] = 0
                continue

            deceleration, speed_before, speed_after = estimate
            qualifies = (
                deceleration >= self.decel_thresh_bl_s2 and speed_before >= self.min_speed_before_bl_s
            )

            if qualifies:
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
                    "decel_thresh_bl_s2": self.decel_thresh_bl_s2,
                    "window_s": self.window_s,
                    "min_speed_before_bl_s": self.min_speed_before_bl_s,
                    "observed_decel_bl_s2": round(deceleration, 4),
                    "speed_before_bl_s": round(speed_before, 4),
                    "speed_after_bl_s": round(speed_after, 4),
                    "streak_frames": streak,
                    "threshold_status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
                }
            )
            self.logger.info(
                "sudden_decel | track=%s decel=%.4f bl/s^2 (thresh %.4f) before=%.4f after=%.4f",
                track_id,
                deceleration,
                self.decel_thresh_bl_s2,
                speed_before,
                speed_after,
            )
            return RuleHit(
                track_ids=[track_id],
                zone_id=None,
                confidence=float(track.mean_confidence),
                rule_params=rule_params,
                reason=f"decel {deceleration:.4f} bl/s^2 >= {self.decel_thresh_bl_s2}",
            )

        return None

    def reset(self) -> None:
        """Clear debounce and streak state."""
        super().reset()
        self._streaks.clear()
