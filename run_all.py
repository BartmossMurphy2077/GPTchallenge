"""
Runner: loads all activity fragments in order via exec(..., globals()).
Mirrors the pattern described in the activity sheet (99_loader_example.py).
"""
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

g = globals()

for fname in [
    "00_setup.py",
    "01_batching.py",
    "02_core_modules.py",
    "03_models_bert_bart.py",
    "04_model_gpt_skeleton.py",
    "05_training_utils_and_demos.py",
]:
    print(f"\n{'='*60}\nLoading {fname}\n{'='*60}")
    with open(fname) as f:
        exec(f.read(), g)
