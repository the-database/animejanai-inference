# Native kernels per GPU

Results of `bench.sh` on each GPU: every shipped model the native kernels cover, native=off
(TensorRT) then on, same build, at libaji level, with the outputs compared. One `<gpu>.txt` per
card (the script's output as it ran), summarised in `../README.md`.

To run on a Linux box (Ubuntu 24.04 with the CUDA toolkit, e.g. Runpod's PyTorch template): copy
`setup.sh`, `bench.sh` and the upload bundle (video models, clips, the manga page, confs, the TFDAT
v2 exporter) to `R` (default `/workspace`), run `setup.sh`, then `bench.sh`. The image models come
from the MangaJaNai / IllustrationJaNai releases, TensorRT 11.4 and cudart from NVIDIA's repos.
