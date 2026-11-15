import json, sys, os, ast, collections, re
d, tag = sys.argv[1], sys.argv[2]
P = {r["name"]: r.get("averageMs", 0) for r in json.load(open(os.path.join(d, "profile_%s.json" % tag))) if isinstance(r, dict) and "name" in r}
L = json.load(open(os.path.join(d, "layers_%s.json" % tag))); L = L["Layers"] if isinstance(L, dict) else L
def lst(x): return x if isinstance(x, list) else ast.literal_eval(x)
fl = 0.0; ms = 0.0; other = 0.0; rows = collections.defaultdict(lambda: [0.0, 0.0, 0])
for l in L:
    if not isinstance(l, dict): continue
    t = P.get(l.get("Name"), 0)
    if "fprop" in (l.get("TacticName") or "") or "dgrad" in (l.get("TacticName") or ""):
        od = lst(l["Outputs"])[0]["Dimensions"]; w = lst(l["Constants"])[0]["Dimensions"]
        f = 2.0 * od[2] * od[3] * w[0] * w[1] * w[2] * w[3]
        fl += f; ms += t
        k = "%dx%d %d->%d" % (od[3], od[2], w[1], w[0]); rows[k][0] += f; rows[k][1] += t; rows[k][2] += 1
    else:
        other += t
print("%s: convs %.3f TFLOP in %.2f ms = %.0f TFLOPS; non-conv %.2f ms" % (tag, fl/1e12, ms, fl/ms/1e9, other))
for k, v in sorted(rows.items(), key=lambda x: -x[1][1])[:8]: print("   %-22s x%-2d %.2f ms %5.0f TFLOPS" % (k, v[2], v[1], v[0]/v[1]/1e9 if v[1] else 0))
