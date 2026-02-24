#!/usr/bin/env python3
"""
Generate temporary question files for mt2.qmd
Run this before rendering mt2.qmd

This is very similar to mt1/generate_questions.py
"""

import os
import glob
import random
import numpy as np

# Configuration flags - should match mt1.qmd
ALL = True
SOL = True
#SOL = False
QLABELS = True
EXAM = False
#EXAM = True

if EXAM:
    ALL = False
    SOL = False
    QLABELS = False

# Set random seed for reproducibility
my_seed = 48
random.seed(my_seed)
np.random.seed(my_seed)

def q_setup(n):
    """
    Setup question files for category n.

    Creates temporary .qmd files for each question in category n,
    adding question labels and solution includes, then concatenates them.

    Parameters:
    -----------
    n : int
        Question category number (1-6)

    Returns:
    --------
    str : Path to concatenated temporary file (tmpN.qmd)
    """
    # Find all question directories for this category
    q_dirs = sorted(glob.glob(f"Q{n}-*"))
    cat_file = f"tmp{n}.qmd"

    # Create temporary files for each question directory
    for q_dir in q_dirs:
        q_file = os.path.join(q_dir, "q.qmd")
        sol_file = os.path.join(q_dir, "sol.qmd")
        q_tmp = os.path.join(q_dir, "tmp.qmd")

        # Create temporary file with question label, content, and solution
        with open(q_tmp, 'w') as f:
            # Add question label block
            results_option = 'asis' if QLABELS else 'false'
            f.write(f"```{{python}}\n")
            f.write(f"#| echo: false\n")
            f.write(f"#| output: {results_option}\n")
            f.write(f"my_dir = '{q_dir}'\n")
            f.write(f"print(f'**{{my_dir}}.**')\n")
            f.write(f"```\n\n")

            # Append question content
            with open(q_file, 'r') as qf:
                f.write(qf.read())

            # Add solution block
            if SOL:
                with open(sol_file, 'r') as qf:
                    f.write(qf.read())
                    
    # Concatenate all question files
    q_files = [os.path.join(d, "tmp.qmd") for d in q_dirs]
    if not ALL:
        q_files = [random.choice(q_files)]

    with open(cat_file, 'w') as cf:
        for qf in q_files:
            with open(qf, 'r') as f:
                cf.write(f.read())

    return cat_file

if __name__ == "__main__":
    print("Generating question files...")
    for n in range(1, 7):
        cat_file = q_setup(n)
        print(f"  Generated {cat_file}")
    print("Done! Now run: quarto render mt2.qmd")
