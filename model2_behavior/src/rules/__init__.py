"""Rule registry and construction.

Rules are constructed from config and validated against camera geometry. A rule
whose required geometry is missing REFUSES TO CONSTRUCT with a clear error,
unless the caller explicitly opts into skipping it.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from model2_behavior.src.config import Model2Config, RuleConfig
from model2_behavior.src.geometry import CameraGeometry
from model2_behavior.src.logging_setup import get_logger
from model2_behavior.src.rules.base import Rule, RuleGeometryError
from model2_behavior.src.rules.collision_proxy import CollisionProxyRule
from model2_behavior.src.rules.stopped_vehicle import StoppedVehicleRule
from model2_behavior.src.rules.sudden_decel import SuddenDecelRule
from model2_behavior.src.rules.wrong_way import WrongWayRule

logger = get_logger(__name__)

RULE_CLASSES: Dict[str, Type[Rule]] = {
    StoppedVehicleRule.name: StoppedVehicleRule,
    WrongWayRule.name: WrongWayRule,
    SuddenDecelRule.name: SuddenDecelRule,
    CollisionProxyRule.name: CollisionProxyRule,
}


def build_rule(name: str, rule_config: RuleConfig, fps: float) -> Rule:
    """Construct a single rule by name.

    Args:
        name: Rule name; must exist in :data:`RULE_CLASSES`.
        rule_config: Resolved configuration for the rule.
        fps: Measured frames per second.

    Returns:
        The constructed rule.

    Raises:
        KeyError: If the rule name is unknown.
    """
    if name not in RULE_CLASSES:
        raise KeyError(f"Unknown rule {name!r}; known rules: {sorted(RULE_CLASSES)}")
    return RULE_CLASSES[name](rule_config, fps)


def build_rules(
    config: Model2Config,
    fps: float,
    geometry: CameraGeometry,
    strict: bool = True,
) -> List[Rule]:
    """Construct all enabled rules, validating required geometry.

    Args:
        config: Resolved Model 2 configuration.
        fps: Measured frames per second.
        geometry: Camera geometry for the stream.
        strict: When True (default), a rule missing required geometry raises.
            When False, such rules are skipped with a warning (useful for
            running a partial rule set explicitly).

    Returns:
        List of constructed rules, in config order.

    Raises:
        RuleGeometryError: If a rule lacks required geometry and ``strict`` is True.
    """
    rules: List[Rule] = []
    for name, rule_config in config.rules.items():
        if not rule_config.enabled:
            logger.info("Rule disabled by config, skipping | %s", name)
            continue

        rule = build_rule(name, rule_config, fps)
        try:
            rule.check_geometry(geometry)
        except RuleGeometryError as exc:
            if strict:
                raise
            logger.warning("Skipping rule (missing geometry) | %s | %s", name, exc)
            continue

        logger.info(
            "Rule constructed | %s | min_consecutive_frames=%s debounce_frames=%s",
            name,
            rule.min_consecutive_frames,
            rule.debounce_frames,
        )
        rules.append(rule)
    return rules


def describe_rules(rules: List[Rule]) -> List[Dict[str, object]]:
    """Return a serialisable description of the constructed rules.

    Args:
        rules: Constructed rules.

    Returns:
        List of dicts with name, thresholds, and frame conversions.
    """
    described: List[Dict[str, object]] = []
    for rule in rules:
        entry: Dict[str, object] = {
            "name": rule.name,
            "min_consecutive_s": rule.config.min_consecutive_s,
            "min_consecutive_frames": rule.min_consecutive_frames,
            "debounce_s": rule.config.debounce_s,
            "debounce_frames": rule.debounce_frames,
            "severity": rule.config.severity,
            "required_geometry": list(rule.required_geometry),
        }
        entry.update(rule.config.params)
        described.append(entry)
    return described


def get_rule(rules: List[Rule], name: str) -> Optional[Rule]:
    """Return a constructed rule by name, or None."""
    for rule in rules:
        if rule.name == name:
            return rule
    return None
