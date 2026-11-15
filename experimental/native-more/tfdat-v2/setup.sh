#!/bin/bash
# Fetches the TFDAT v2 source (Kim2091/traiNNer-redux-1, branch dis2-tfdat2) and builds the import
# stubs export.py needs (spandrel and the traiNNer registry; only UniUpsampleV3 from arch_util).
# usage: setup.sh  (then: python export.py SCALE CLIP OUTDIR; python ref.py ... for parity)
set -e
cd "$(dirname "$0")"
R=repos/Kim2091/traiNNer-redux-1/contents/traiNNer/archs; Q='?ref=dis2-tfdat2'
mkdir -p stub/traiNNer/archs stub/traiNNer/utils stub/spandrel/__helpers stub/spandrel/util
touch stub/traiNNer/__init__.py stub/traiNNer/archs/__init__.py stub/traiNNer/utils/__init__.py \
      stub/spandrel/__init__.py stub/spandrel/__helpers/__init__.py stub/spandrel/util/__init__.py
gh api "$R/tfdat_arch.py$Q" -H "Accept: application/vnd.github.raw" > stub/traiNNer/archs/tfdat_arch.py
gh api "$R/arch_util.py$Q" -H "Accept: application/vnd.github.raw" > arch_util_full.py
echo 'StateDict = dict' > stub/spandrel/__helpers/model_descriptor.py
printf 'from torch import nn\nclass DropPath(nn.Identity):\n    def __init__(self, *a, **k): super().__init__()\n' > stub/spandrel/util/timm.py
printf 'class _R:\n    def register(self):\n        return lambda f: f\nARCH_REGISTRY = _R()\n' > stub/traiNNer/utils/registry.py
python - <<'PY'
src = open("arch_util_full.py", encoding="utf-8").read()
i = src.index("class UniUpsampleV3")
k = src.find("\nclass ", i + 10)
hdr = ("import math\nfrom typing import Literal\nimport torch\nfrom torch import nn\n"
       'SampleMods3 = Literal["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample", "transpose+conv", "lda", "pa_up"]\n'
       "class DySample(nn.Module):\n    pass\n")
open("stub/traiNNer/archs/arch_util.py", "w", encoding="utf-8").write(hdr + src[i:k if k > 0 else len(src)])
PY
rm arch_util_full.py
echo ok
