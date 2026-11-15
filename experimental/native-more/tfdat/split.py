import json, sys, os, collections, re, ast
d, tag = sys.argv[1], sys.argv[2]
P = {r["name"]: r.get("averageMs", 0) for r in json.load(open(os.path.join(d, "profile_%s.json" % tag))) if isinstance(r, dict) and "name" in r}
L = json.load(open(os.path.join(d, "layers_%s.json" % tag))); L = L["Layers"] if isinstance(L, dict) else L
grp = {"groups": "FDAT body (24 blocks)", "conv_after": "FDAT body (24 blocks)", "upsampler": "upsampler + hr_refine", "hr_refine": "upsampler + hr_refine",
       "flow_net": "alignment (flow, warp, refine)", "flow_net_": "alignment (flow, warp, refine)", "align_refine": "alignment (flow, warp, refine)",
       "align_refine_": "alignment (flow, warp, refine)", "AveragePool": "alignment (flow, warp, refine)", "AveragePool_": "alignment (flow, warp, refine)",
       "Gather_": "alignment (flow, warp, refine)", "enc": "per-frame encoder (enc*)", "temporal_fusion": "temporal fusion + conv_first", "conv_first": "temporal fusion + conv_first"}
t = collections.defaultdict(float); flop = collections.defaultdict(float)
for l in L:
    if not isinstance(l, dict): continue
    md = str(l.get("Metadata", "")) + " " + l.get("Name", "")
    mods = re.findall(r"/model/([A-Za-z_]+)", md)
    g = collections.Counter(grp.get(m, "other") for m in mods).most_common(1)[0][0] if mods else "other (unlabelled myelin)"
    t[g] += P.get(l.get("Name"), 0)
    if "fprop" in (l.get("TacticName") or "") or "gemm" in (l.get("TacticName") or ""):
        try:
            ins = ast.literal_eval(l["Inputs"]); outs = ast.literal_eval(l["Outputs"]); cs = ast.literal_eval(l.get("Constants", "[]"))
            od = outs[0]["Dimensions"]
            if "fprop" in l["TacticName"] and cs:
                w = cs[0]["Dimensions"]  # [cout, cin, kh, kw]
                flop[g] += 2.0 * od[0] * od[2] * od[3] * w[0] * w[1] * w[2] * w[3]
            elif len(ins) >= 1:
                a = ins[0]["Dimensions"]; k = a[-1]
                n = 1
                for x in od: n *= x
                flop[g] += 2.0 * n * k
        except Exception as e:
            pass
tot = sum(t.values())
print("%s: total %.2f ms" % (tag, tot))
for k, v in sorted(t.items(), key=lambda x: -x[1]):
    print("  %-34s %7.2f ms %5.1f%%   %6.3f TFLOP%s" % (k, v, 100*v/tot, flop[k]/1e12, ("  (%.0f TFLOPS)" % (flop[k]/v/1e9) if v and flop[k] else "")))
