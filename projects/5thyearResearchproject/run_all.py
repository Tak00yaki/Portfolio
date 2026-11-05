"""Run every week's script in order; stop at the first one that fails.

Weeks 5-6 showed why: a broken 2nd_week.py went unnoticed because only the current week
was run. This runs them all, then lists every saved result and figure that changed, so a
change to shared code that moves an older week's numbers is seen before a commit.

Usage: python run_all.py           (RL weeks only re-analyse their saved runs)
       python run_all.py --train   (RL weeks also train any run that is missing)
       python run_all.py 7 11      (only the scripts whose names start with these numbers)
"""

import glob
import hashlib
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["1st_week.py", "2nd_week.py", "3rd_4th_week.py", "5th_6th_week.py", "7th_8th_week.py",
           "9th_10th_week.py", "11th_week.py", "12th_week.py", "13th_14th_week.py"]
RL = {"12th_week.py", "13th_14th_week.py"}      # take "analyse" unless --train
WATCH = ["results/*.json", "results/*.csv", "results/rl/*.json", "results/rl/*.csv", "figures/*.png"]


def snapshot():
    out = {}
    for pattern in WATCH:
        for path in glob.glob(os.path.join(HERE, pattern)):
            with open(path, "rb") as f:
                out[os.path.relpath(path, HERE)] = hashlib.sha256(f.read()).hexdigest()
    return out


def main(args):
    train = "--train" in args
    only = [a for a in args if not a.startswith("--")]
    scripts = [s for s in SCRIPTS if not only or any(re.match(rf"{n}\D", s) for n in only)]
    missing = [s for s in SCRIPTS if not os.path.exists(os.path.join(HERE, s))]
    if missing:
        sys.exit(f"missing scripts: {', '.join(missing)}")
    before = snapshot()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    times = {}
    for script in scripts:
        cmd = [sys.executable, script] + (["analyse"] if script in RL and not train else [])
        log = os.path.join(HERE, "logs", script.replace(".py", ".log"))
        print(f"{' '.join(cmd[1:]):<28}", end=" ", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            code = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        times[script] = time.time() - t0
        print(f"{'ok' if code == 0 else 'FAILED (exit ' + str(code) + ')'}  {times[script]:6.0f} s   log: logs/{os.path.basename(log)}")
        if code:
            with open(log) as f:
                print("".join(f.readlines()[-20:]))
            sys.exit(1)
    after = snapshot()
    changed = sorted(p for p in after if before.get(p) != after[p])
    gone = sorted(set(before) - set(after))
    print(f"\nAll {len(scripts)} scripts ran in {sum(times.values()) / 60:.0f} min.")
    if changed or gone:
        print("Results or figures that changed (check these are meant to before committing):")
        for p in changed:
            print(f"  {'new' if p not in before else 'changed'}  {p}")
        for p in gone:
            print(f"  removed  {p}")
    else:
        print("No saved result or figure changed.")


if __name__ == "__main__":
    main(sys.argv[1:])
