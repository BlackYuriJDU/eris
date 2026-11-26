"""PROTOTYPE for #85, throwaway: what each config's dump changes against the baseline dump. Usage: diff.py logs/"""
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
base = json.loads((d / "dump_base.json").read_text())
for cfg in ("hub", "entry", "hub+entry"):
    other = json.loads((d / f"dump_{cfg}.json").read_text())
    print(f"=== {cfg} vs base")
    for run in (k for k in base if "/" in k):
        for kind in sorted(set(base[run]) | set(other[run])):
            b, o = set(base[run].get(kind, [])), set(other[run].get(kind, []))
            if b != o:
                print(f"  {run} {kind}: -{len(b - o)} +{len(o - b)}")
                for x in sorted(b - o): print(f"     - {x}")
                for x in sorted(o - b): print(f"     + {x}")
