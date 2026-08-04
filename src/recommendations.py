"""Actionable cybersecurity recommendations for detected network attacks."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Build structured security recommendations based on attack type and risk score."""

    def __init__(self) -> None:
        """Initialize the recommendation engine with a default catalog of attack guidance."""
        self._recommendation_catalog: dict[str, dict[str, Any]] = {
            "Normal": {
                "severity": "Low",
                "description": "No suspicious activity detected. Continue routine monitoring.",
                "actions": [
                    "Maintain normal monitoring and logging.",
                    "Validate that baseline network behavior remains stable.",
                ],
                "immediate_steps": [
                    "Continue standard observation and incident review.",
                ],
            },
            "DDoS": {
                "severity": "Critical",
                "description": "Distributed denial-of-service activity is overwhelming the target service.",
                "actions": [
                    "Activate rate limiting and traffic scrubbing controls.",
                    "Block or challenge suspicious source ranges.",
                    "Scale capacity or divert traffic to mitigation services.",
                ],
                "immediate_steps": [
                    "Isolate affected services and alert the SOC team.",
                    "Review firewall and CDN mitigation logs immediately.",
                ],
            },
            "DoS": {
                "severity": "High",
                "description": "A denial-of-service condition is affecting service availability.",
                "actions": [
                    "Throttle excessive requests and enforce connection limits.",
                    "Investigate burst traffic patterns and source reputation.",
                    "Enable service protection controls if available.",
                ],
                "immediate_steps": [
                    "Suspend suspicious sessions and preserve logs.",
                    "Notify impacted service owners and validate availability.",
                ],
            },
            "Port Scan": {
                "severity": "High",
                "description": "Reconnaissance activity suggests an attacker is probing exposed services.",
                "actions": [
                    "Review scanning source history and block repeated probes.",
                    "Restrict inbound access to only required services.",
                    "Inspect exposed systems for unauthorized access paths.",
                ],
                "immediate_steps": [
                    "Alert the SOC and investigate the source host.",
                    "Confirm whether any services were enumerated successfully.",
                ],
            },
            "Botnet": {
                "severity": "Critical",
                "description": "Compromised hosts are participating in coordinated malicious activity.",
                "actions": [
                    "Contain infected endpoints and isolate them from the network.",
                    "Collect forensic artifacts and review command-and-control traffic.",
                    "Reset credentials and block malicious infrastructure.",
                ],
                "immediate_steps": [
                    "Quarantine affected hosts and preserve evidence.",
                    "Engage endpoint detection and response teams.",
                ],
            },
            "Brute Force": {
                "severity": "High",
                "description": "Repeated authentication attempts suggest credential stuffing or password guessing.",
                "actions": [
                    "Enable account lockout and multifactor authentication.",
                    "Review failed login patterns and identify targeted accounts.",
                    "Reset compromised credentials and inspect access logs.",
                ],
                "immediate_steps": [
                    "Block the source IP and alert the identity team.",
                    "Review privileged account access for suspicious activity.",
                ],
            },
            "SQL Injection": {
                "severity": "Critical",
                "description": "An attempted or successful SQL injection indicates a high-risk application compromise.",
                "actions": [
                    "Contain the affected application and review server logs.",
                    "Validate database integrity and inspect for unauthorized queries.",
                    "Patch vulnerable SQL input handling and review access controls.",
                ],
                "immediate_steps": [
                    "Isolate the application and preserve evidence.",
                    "Notify application owners and incident response personnel.",
                ],
            },
            "XSS": {
                "severity": "High",
                "description": "Cross-site scripting activity may indicate client-side exploitation attempts.",
                "actions": [
                    "Review web application logs and identify the affected endpoint.",
                    "Apply input sanitization and content security controls.",
                    "Investigate whether malicious scripts executed in the browser.",
                ],
                "immediate_steps": [
                    "Block the associated request patterns and alert the web team.",
                    "Validate browser-side compromise indicators.",
                ],
            },
            "Infiltration": {
                "severity": "Critical",
                "description": "Unauthorized access activity suggests a potential foothold inside the environment.",
                "actions": [
                    "Isolate affected systems and review host telemetry.",
                    "Revoke suspicious access and rotate credentials.",
                    "Investigate persistence mechanisms and lateral movement.",
                ],
                "immediate_steps": [
                    "Contain the host and engage incident response.",
                    "Collect endpoint and network evidence immediately.",
                ],
            },
            "Web Attack": {
                "severity": "High",
                "description": "Web-layer exploitation activity is affecting application security.",
                "actions": [
                    "Inspect web server logs and WAF events for the attack pattern.",
                    "Apply web application security controls and patches.",
                    "Verify whether exploitation was successful or blocked.",
                ],
                "immediate_steps": [
                    "Block malicious IPs and preserve logs.",
                    "Notify the application support team promptly.",
                ],
            },
            "Heartbleed": {
                "severity": "Critical",
                "description": "The Heartbleed vulnerability may have exposed sensitive information in TLS traffic.",
                "actions": [
                    "Patch vulnerable services and rotate certificates and secrets.",
                    "Review for data exposure and unauthorized access.",
                    "Validate system integrity and inspect memory exposure indicators.",
                ],
                "immediate_steps": [
                    "Contain impacted services and initiate incident handling.",
                    "Assess whether sensitive data may have been exposed.",
                ],
            },
            "FTP Patator": {
                "severity": "High",
                "description": "Brute-force activity against FTP services indicates credential abuse attempts.",
                "actions": [
                    "Disable or lock affected FTP accounts and review authentication logs.",
                    "Enforce stronger password policies and MFA where possible.",
                    "Inspect the service for unauthorized access attempts.",
                ],
                "immediate_steps": [
                    "Block the source and alert the infrastructure team.",
                    "Review whether any successful logins occurred.",
                ],
            },
            "SSH Patator": {
                "severity": "High",
                "description": "Repeated SSH authentication attempts indicate a brute-force or credential abuse campaign.",
                "actions": [
                    "Review SSH authentication logs and disable affected accounts if needed.",
                    "Enforce SSH hardening and multifactor authentication.",
                    "Investigate whether the attacker gained access to a host.",
                ],
                "immediate_steps": [
                    "Block the source and notify the administrators.",
                    "Validate whether any privileged sessions were established.",
                ],
            },
        }

    def get_recommendation(self, attack_type: str, risk_score: float) -> dict[str, Any]:
        """Return a recommendation payload for the given attack type and risk score."""
        if not isinstance(attack_type, str) or not attack_type.strip():
            raise ValueError("Attack type must be a non-empty string")
        if not isinstance(risk_score, (int, float)):
            raise ValueError("Risk score must be numeric")

        normalized_attack_type = self._normalize_attack_type(attack_type)
        catalog_entry = self._recommendation_catalog.get(normalized_attack_type)
        if catalog_entry is None:
            logger.warning("No specific recommendation found for attack type '%s'; using default", attack_type)
            catalog_entry = self._default_recommendation()

        severity = self._adjust_severity(catalog_entry["severity"], risk_score)
        description = catalog_entry["description"]
        actions = list(catalog_entry["actions"])
        immediate_steps = list(catalog_entry["immediate_steps"])

        return {
            "severity": severity,
            "description": description,
            "actions": actions,
            "immediate_steps": immediate_steps,
        }

    def add_attack_type(self, attack_type: str, recommendation: dict[str, Any]) -> None:
        """Register a new attack type with its recommendation payload."""
        if not isinstance(attack_type, str) or not attack_type.strip():
            raise ValueError("Attack type must be a non-empty string")
        if not isinstance(recommendation, dict):
            raise TypeError("Recommendation payload must be a dictionary")

        required_keys = {"severity", "description", "actions", "immediate_steps"}
        missing_keys = required_keys.difference(recommendation.keys())
        if missing_keys:
            raise ValueError(f"Recommendation payload is missing required keys: {', '.join(sorted(missing_keys))}")

        self._recommendation_catalog[self._normalize_attack_type(attack_type)] = recommendation
        logger.info("Added recommendation entry for attack type '%s'", attack_type)

    def _normalize_attack_type(self, attack_type: str) -> str:
        """Normalize attack names to a consistent catalog key."""
        return attack_type.strip().title()

    def _adjust_severity(self, base_severity: str, risk_score: float) -> str:
        """Escalate severity when the risk score exceeds known thresholds."""
        severity_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        severity = base_severity
        if risk_score >= 0.9:
            severity = "Critical"
        elif risk_score >= 0.7:
            severity = "High"
        elif risk_score >= 0.4:
            severity = "Medium"

        current_rank = severity_rank.get(severity, 1)
        base_rank = severity_rank.get(base_severity, 1)
        return severity if current_rank >= base_rank else base_severity

    def _default_recommendation(self) -> dict[str, Any]:
        """Return a safe fallback recommendation for unknown attack types."""
        return {
            "severity": "Medium",
            "description": "Unexpected or unclassified network activity has been detected.",
            "actions": [
                "Investigate the event and collect related telemetry.",
                "Review affected systems, logs, and network flows.",
                "Escalate to the incident response team if the behavior is suspicious.",
            ],
            "immediate_steps": [
                "Preserve relevant logs and isolate the impacted assets if necessary.",
                "Validate the activity against known threat intel and baseline behavior.",
            ],
        }
