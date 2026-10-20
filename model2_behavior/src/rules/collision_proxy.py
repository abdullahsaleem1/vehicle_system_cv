"""Rule 4: collision proxy.

Fires when ALL of the following hold:
    1. Two tracks' boxes overlap with IoU >= ``iou_thresh``.
    2. The overlap is SUSTAINED for at least ``min_overlap_s`` seconds
       (a single overlapping frame is not a collision).
    3. Both tracks were MOVING before contact (speed >=
       ``min_pre_contact_speed_bl_s``) when ``both_moving_before`` is set.
    4. At least one track then decelerates sharply or stops within ``window_s``
       (speed drop >= ``post_contact_speed_drop_bl_s``).

A pair is debounced together using a synthetic key derived from the two track
ids, so one collision produces one alert, not two.

This is a PROXY, not a physically validated collision detector: it uses box
overlap and kinematics rather than contact forces.

THRESHOLD STATUS: ALL thresholds are PLACEHOLDER, unvalidated until the Step 8
evaluation harness reports per-rule precision/recall.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from model2_behavior.src.rules.base import FrameContext, Rule, RuleHit
from model2_behavior.src.tracks import TrackState, boxes_iou


class CollisionProxyRule(Rule):
    """Detects sustained box overlap followed by a sharp slowdown."""

    name = "collision_proxy"
    required_geometry = ()

    def __init__(self, config, fps: float) -> None:
        """Initialise and read rule thresholds.

        Args:
            config: Resolved rule configuration.
            fps: Measured frames per second.
        """
        super().__init__(config, fps)
        params = config.params
        self.iou_thresh = float(params.get("iou_thresh", 0.30))
        self.min_overlap_s = float(params.get("min_overlap_s", 0.3))
        self.both_moving_before = bool(params.get("both_moving_before", True))
        self.min_pre_contact_speed_bl_s = float(params.get("min_pre_contact_speed_bl_s", 0.20))
        self.post_contact_speed_drop_bl_s = float(params.get("post_contact_speed_drop_bl_s", 0.25))
        self.window_s = float(params.get("window_s", 1.5))
        self.speed_window_s = float(params.get("speed_window_s", 0.5))

        self.min_overlap_frames = self.seconds_to_frames(self.min_overlap_s)
        # pair key -> consecutive overlapping frames
        self._overlap_streaks: Dict[Tuple[int, int], int] = {}
        # pair key -> speed snapshot captured during overlap (pre-contact)
        self._contact_speeds: Dict[Tuple[int, int], Dict[int, float]] = {}

    @staticmethod
    def _pair_key(track_a: int, track_b: int) -> Tuple[int, int]:
        """Return an order-independent key for a track pair."""
        return (track_a, track_b) if track_a <= track_b else (track_b, track_a)

    @staticmethod
    def _pair_debounce_key(pair: Tuple[int, int]) -> int:
        """Return a single int debounce key for a pair.

        Debounce storage is keyed by int, so the two ids are combined with a
        Cantor-style pairing that stays collision-free for non-negative ids.
        """
        first, second = pair
        return (first + second) * (first + second + 1) // 2 + second

    def update(self, context: FrameContext, tracks: List[TrackState]) -> Optional[RuleHit]:
        """Evaluate collision-proxy conditions for this frame.

        Args:
            context: Frame context.
            tracks: Active tracks.

        Returns:
            A hit when a qualifying overlapping pair is found.
        """
        pair_speeds: Dict[int, Optional[float]] = {
            track.track_id: track.speed_bl_s(self.speed_window_s) for track in tracks
        }
        by_id = {track.track_id: track for track in tracks}
        seen_pairs: Set[Tuple[int, int]] = set()

        for track_a, track_b in combinations(sorted(by_id), 2):
            pair = self._pair_key(track_a, track_b)
            seen_pairs.add(pair)

            box_a = by_id[track_a].box_at_latest()
            box_b = by_id[track_b].box_at_latest()
            if box_a is None or box_b is None:
                continue

            iou = boxes_iou(box_a, box_b)
            if iou < self.iou_thresh:
                self._overlap_streaks[pair] = 0
                self._contact_speeds.pop(pair, None)
                continue

            # Record pre-contact speeds the first time this pair starts overlapping.
            if pair not in self._contact_speeds:
                self._contact_speeds[pair] = {
                    track_id: speed
                    for track_id, speed in (
                        (track_a, pair_speeds.get(track_a)),
                        (track_b, pair_speeds.get(track_b)),
                    )
                    if speed is not None
                }

            self._overlap_streaks[pair] = self._overlap_streaks.get(pair, 0) + 1
            streak = self._overlap_streaks[pair]
            if streak < self.min_overlap_frames:
                continue

            pre_speeds = self._contact_speeds.get(pair, {})
            # Condition 3: both tracks must have been moving before contact.
            if self.both_moving_before:
                moving = [
                    speed for speed in pre_speeds.values() if speed >= self.min_pre_contact_speed_bl_s
                ]
                if len(moving) < 2:
                    continue

            # Condition 4: at least one track dropped sharply relative to pre-contact.
            dropping_tracks: List[int] = []
            for track_id in (track_a, track_b):
                before = pre_speeds.get(track_id)
                after = pair_speeds.get(track_id)
                if before is None or after is None:
                    continue
                if (before - after) >= self.post_contact_speed_drop_bl_s:
                    dropping_tracks.append(track_id)

            if not dropping_tracks:
                continue

            debounce_key = self._pair_debounce_key(pair)
            if self.is_debounced(debounce_key, context.frame_index):
                continue

            self.arm_debounce(debounce_key, context.frame_index)
            self._overlap_streaks[pair] = 0
            self._contact_speeds.pop(pair, None)

            involved = sorted((track_a, track_b))
            mean_conf = min(by_id[track_id].mean_confidence for track_id in involved)

            rule_params = self.base_rule_params()
            rule_params.update(
                {
                    "iou_thresh": self.iou_thresh,
                    "min_overlap_s": self.min_overlap_s,
                    "min_overlap_frames": self.min_overlap_frames,
                    "both_moving_before": self.both_moving_before,
                    "min_pre_contact_speed_bl_s": self.min_pre_contact_speed_bl_s,
                    "post_contact_speed_drop_bl_s": self.post_contact_speed_drop_bl_s,
                    "window_s": self.window_s,
                    "speed_window_s": self.speed_window_s,
                    "observed_iou": round(iou, 4),
                    "overlap_frames": streak,
                    "pre_contact_speeds_bl_s": {str(k): round(v, 4) for k, v in pre_speeds.items()},
                    "dropping_track_ids": dropping_tracks,
                    "threshold_status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
                }
            )
            self.logger.info(
                "collision_proxy | tracks=%s iou=%.4f overlap_frames=%s dropping=%s",
                involved,
                iou,
                streak,
                dropping_tracks,
            )
            return RuleHit(
                track_ids=involved,
                zone_id=None,
                confidence=float(mean_conf),
                rule_params=rule_params,
                reason=f"IoU {iou:.3f} sustained {streak} frames; {dropping_tracks} decelerated",
            )

        # Drop streaks for pairs no longer present together.
        for pair in list(self._overlap_streaks):
            if pair not in seen_pairs:
                self._overlap_streaks.pop(pair, None)
                self._contact_speeds.pop(pair, None)

        return None

    def reset(self) -> None:
        """Clear debounce, streak, and contact state."""
        super().reset()
        self._overlap_streaks.clear()
        self._contact_speeds.clear()
