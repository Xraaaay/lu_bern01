# -*- coding: utf-8 -*-
"""
Created on 2026-09-12
@author: Ruowen Xiao
"""

import numpy as np
import json
from pathlib import Path

def save_json(path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def save_ndarray(path, ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, ndarray)

