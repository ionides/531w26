#!/usr/bin/env python3
"""
Generate temporary question files for class14.qmd
Run this before rendering the qmd to pdf
"""

Q = "mt2/Q1-01"

import os
import glob
import random
import numpy as np

# Set random seed for reproducibility
random.seed(117)
np.random.seed(117)

q_file = os.path.join("../", Q, "q.qmd")
q_tmp = "tmp14.qmd"

with open(q_tmp, 'w') as f:
    with open(q_file, 'r') as qf:
        f.write(qf.read())



