#!/usr/bin/env python3
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class Manifest:
    inputs: Dict[str, Any]
    params: Dict[str, Any]
    code_sha: str
    checksums: Dict[str, str]
    signature: str = ""

    def sign(self, secret: str) -> None:
        payload = json.dumps({k:v for k,v in asdict(self).items() if k != 'signature'}, sort_keys=True)
        self.signature = hashlib.sha256((payload + secret).encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
