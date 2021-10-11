import os
import glob
import re
import pandas as pd
from pymatgen.io.lammps.outputs import parse_lammps_log

def concat_log(log_pattern, step=None, working_dir=None):
    # based on pymatgen parse lammps dumps func
    working_dir = working_dir or os.getcwd()
    log_files = f"{working_dir}/{log_pattern}"
    files = glob.glob(log_files)
    if len(files) > 1:
        pattern = r"%s" % log_pattern.replace("*", "([0-9]+)")
        pattern = '.*' + pattern.replace("\\", "\\\\")
        files = sorted(files, key=lambda f: int(re.match(pattern, f).group(1)))
    logs = []
    for file in files:
        logs.append(parse_lammps_log(file)[0])
    for p, l in enumerate(logs[:-1]):
        logs[p] = l[:-1]
    full_log = pd.concat(logs, ignore_index=True)
    if step:
        # take every N step to speed the code
        full_log = full_log.loc[range(1, full_log.shape[0], 50000)]
    return full_log





