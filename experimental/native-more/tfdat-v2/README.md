# TFDAT v2 test models

Random-weight TFDAT v2 exports at 1x, 2x and 4x for the native parity checks: `setup.sh` fetches the
v2 source and builds import stubs, `export.py SCALE CLIP OUTDIR` writes an fp32 and an fp16 ONNX
named like traiNNer's exports (`/model/...`), with every zero-initialised head made non-zero, and
`ref.py` runs a model through `aji_native_vid_run` and ONNX Runtime (CPU, fp32 export) on real
frames and prints the PSNR. Needs PyTorch, onnx, onnxruntime and numpy.
