import re
import subprocess
from pathlib import Path
import pandas as pd

GEM5 = "./gem5.fast"
CONFIG = "config_ibilinear.py"
KERNELS = ["scalar", "u1v", "u2v", "u1v_pack", "u2v_pack", "u2v_inv", "stride"]
ITERS = 20  # should match iter in bench_ibilinear_gem5
BASELINE = "scalar"

KEYS = {
    r"\.numCycles\b": "cycles",
    r"\.committedInsts\b": "insts",
    r"\.ipc\b": "ipc",
    r"(?:dcache|l1d[\w-]*)\.overallMissRate::total\b": "l1d_missrate",
    r"(?:dcache|l1d[\w-]*)\.overallAvgMissLatency::total\b": "l1d_misslat",
}


def run(kernel):
    outdir = f"m5out_{kernel}"
    r = subprocess.run(
        [GEM5, f"--outdir={outdir}", CONFIG, kernel], capture_output=True, text=True
    )
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit(f"gem5 KO sur {kernel}")
    txt = Path(outdir, "stats.txt").read_text()
    row = {"kernel": kernel}
    for pat, col in KEYS.items():
        m = re.search(rf"^\S*{pat}\s+([\d.eE+-]+)", txt, re.M)
        row[col] = float(m.group(1)) if m else float("nan")
    return row


df = pd.DataFrame(run(k) for k in KERNELS).set_index("kernel")
df["cycles_per_iter"] = df["cycles"] / ITERS
df["speedup_vs_scalar"] = df.loc[BASELINE, "cycles"] / df["cycles"]
df["l1d_misslat_cycles"] = df["l1d_misslat"] / 1000  # 1 cycle = 1000 ticks @1GHz

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print(
    df[
        [
            "cycles",
            "cycles_per_iter",
            "ipc",
            "l1d_missrate",
            "l1d_misslat",
            "speedup_vs_scalar",
        ]
    ]
)
df.to_csv("bench_results.csv")
print("\n-> bench_results.csv")
