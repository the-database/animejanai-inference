import json, sys, os, collections, re
d = sys.argv[1]
P = [r for r in json.load(open(os.path.join(d, "profile.json"))) if isinstance(r, dict) and "name" in r]
L = json.load(open(os.path.join(d, "layers.json"))); L = L["Layers"] if isinstance(L, dict) else L
info = {l.get("Name"): l for l in L if isinstance(l, dict)}
cat = collections.defaultdict(float); n = collections.Counter(); ex = {}
for r in P:
    l = info.get(r["name"], {}); t = (l.get("TacticName") or "") ; ty = l.get("LayerType", "")
    nm = r["name"]
    if "fprop" in t or "conv" in t.lower(): k = "conv (xmma)"
    elif re.search(r"gemm|xmma|cutlass|matmul|Gemm|sgemm|hgemm", t): k = "gemm"
    elif "fmha" in t.lower() or "mha" in t.lower() or "attention" in t.lower(): k = "fused attention"
    elif t.startswith("__myl"): k = "myelin " + re.sub(r"_0x[0-9a-f]+.*", "", t)[6:]
    else: k = ty or t[:40] or "?"
    cat[k] += r.get("averageMs", 0); n[k] += 1; ex.setdefault(k, t[:120])
tot = sum(cat.values())
print("total %.3f ms over %d layers" % (tot, sum(n.values())))
for k, v in sorted(cat.items(), key=lambda x: -x[1])[:25]: print("%7.3f ms %5.1f%% %4d  %-28s %s" % (v, 100*v/tot, n[k], k[:28], ex[k][:90]))
