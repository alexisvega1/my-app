#!/usr/bin/env python3
from __future__ import annotations
import json
from typing import Dict, Any
from prometheus_client import Gauge

VI_TOTAL = Gauge('vi_total', 'Variation of Information total', ['dataset','version'])
PR_PREC = Gauge('syn_pr_precision', 'Synapse precision', ['dataset','version'])
PR_REC = Gauge('syn_pr_recall', 'Synapse recall', ['dataset','version'])
MERGE_ALARM = Gauge('merge_alarm_rate', 'Merge alarm rate', ['dataset','version'])

def generate_report(metrics: Dict[str, Any], dataset: str, version: str) -> Dict[str, str]:
    # Update metrics
    VI_TOTAL.labels(dataset, version).set(metrics.get('vi_total', 0.0))
    PR_PREC.labels(dataset, version).set(metrics.get('precision', 0.0))
    PR_REC.labels(dataset, version).set(metrics.get('recall', 0.0))
    MERGE_ALARM.labels(dataset, version).set(metrics.get('merge_alarm', 0.0))
    # Build HTML/JSON
    html = f"""
    <html><body>
    <h2>QC Report - {dataset}:{version}</h2>
    <ul>
      <li>VI_total: {metrics.get('vi_total',0.0):.4f}</li>
      <li>Synapse P/R: {metrics.get('precision',0.0):.3f} / {metrics.get('recall',0.0):.3f}</li>
      <li>Merge-alarm rate: {metrics.get('merge_alarm',0.0):.3f}</li>
    </ul>
    </body></html>
    """
    return {"html": html, "json": json.dumps(metrics)}
