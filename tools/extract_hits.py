#!/usr/bin/env python3
"""Dump (event, track, layer, chipID, row, col, gs1, gs2, gs3) for the first N events.

    python3 tools/extract_hits.py [N] [input.root] > hits.txt

Feeds tools/check_range_cuts.C, which replays YAlignment's coordinate step on the
same hits. Needs uproot, not ROOT.
"""
import sys
import uproot

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SRC = sys.argv[2] if len(sys.argv) > 2 else "XXXXinput.root"

t = uproot.open(SRC)["DataInput"]
b = "event/Track/Track."
a = t.arrays([b + n for n in ("ChipID[7]", "row[7]", "col[7]", "s1[7]", "s2[7]", "s3[7]")],
             entry_stop=N, library="np")
g = lambda k: a[b + k]
cid, row, col = g("ChipID[7]"), g("row[7]"), g("col[7]")
s1, s2, s3 = g("s1[7]"), g("s2[7]"), g("s3[7]")

for ev in range(len(cid)):
    for it in range(len(cid[ev])):
        for l in range(7):
            if int(cid[ev][it][l]) < 0:
                continue
            print(f"{ev} {it} {l} {int(cid[ev][it][l])} "
                  f"{float(row[ev][it][l]):.6f} {float(col[ev][it][l]):.6f} "
                  f"{float(s1[ev][it][l]):.9f} {float(s2[ev][it][l]):.9f} {float(s3[ev][it][l]):.9f}")
