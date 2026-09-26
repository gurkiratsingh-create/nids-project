"""Dashboard Utilities for Streamlit Session Management and Simulation."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

import numpy as np


def generate_simulated_traffic_stream(num_samples: int = 10) -> List[Dict[str, Any]]:
    """Generate realistic live stream traffic packets for testing SOC UI interactive simulation."""
    np.random.seed(int(datetime.datetime.now().timestamp()) % 10000)
    
    classes = ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "Bot", "FTP-Patator", "Web Attack"]
    probabilities = [0.60, 0.15, 0.10, 0.08, 0.03, 0.02, 0.02]
    
    simulated_packets = []
    for _ in range(num_samples):
        attack = str(np.random.choice(classes, p=probabilities))
        is_attack = attack != "BENIGN"
        
        flow = {
            "Source IP": f"192.168.1.{np.random.randint(2, 254)}" if not is_attack else f"45.33.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}",
            "Destination IP": f"10.0.0.{np.random.randint(2, 254)}",
            "Source Port": int(np.random.randint(1024, 65535)),
            "Destination Port": int(np.random.choice([80, 443, 22, 21, 8080, 53])),
            "Protocol": str(np.random.choice(["TCP", "UDP"], p=[0.85, 0.15])),
            "Flow Duration": float(np.random.exponential(scale=100000 if not is_attack else 500000)),
            "Total Fwd Packets": int(np.random.randint(1, 20 if not is_attack else 500)),
            "Total Bwd Packets": int(np.random.randint(0, 20 if not is_attack else 500)),
            "Flow Bytes/s": float(np.random.exponential(scale=5000 if not is_attack else 950000)),
            "SYN Flag Count": int(np.random.binomial(n=5, p=0.1 if not is_attack else 0.8)),
            "simulated_attack_label": attack,
        }
        simulated_packets.append(flow)
        
    return simulated_packets
