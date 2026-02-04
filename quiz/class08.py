#!/usr/bin/env python3
"""
Generate temporary question files for class08.qmd
Run this before rendering class08.qmd
"""

Q = "Q1-03"

import os
import glob
import random
import numpy as np

# Set random seed for reproducibility
random.seed(117)
np.random.seed(117)

q_file = os.path.join("../mt1/", Q, "q.qmd")
q_tmp = "tmp08.qmd"

with open(q_tmp, 'w') as f:
    with open(q_file, 'r') as qf:
        f.write(qf.read())



