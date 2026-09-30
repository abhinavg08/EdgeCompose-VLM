# Third-party attribution and license scope

This repository has no overall license for its original code and reports. The notices below
preserve upstream terms; they do not relicense third-party work or grant new rights to the
complete project.

## Adapted code

* `edgecompose/compression/visionzip.py` ports the selection and merging logic from
  [VisionZip's Qwen2.5-VL implementation](https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py).
  Upstream notices: Copyright 2025 Senqiao Yang; Copyright 2025 The Qwen Team and The
  HuggingFace Inc. team. All rights reserved. The upstream file and VisionZip repository
  use Apache License 2.0; a copy is in `third_party/APACHE-2.0.txt`.
  This port isolates the compressor from the upstream model fork, retains original M-RoPE
  positions, and computes attention statistics independently per image.
  Reference: Yang et al., *VisionZip: Longer is Better but Not Necessary in Vision Language
  Models*, CVPR 2025 ([paper](https://arxiv.org/abs/2412.04467)).
* `edgecompose/quant/awq_dispatch.py` adapts the forward dispatch from
  [AutoAWQ](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py).
  Copyright (c) 2023 MIT HAN Lab. Upstream MIT license: `third_party/AutoAWQ-MIT.txt`.
  The modification makes the dequantize/GEMM switch threshold configurable; kernels remain
  upstream. AWQ reference: Lin et al., *AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration*, MLSys 2024 ([paper](https://arxiv.org/abs/2306.00978)).

## Model and runtime dependencies

* [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) and its
  [technical report](https://arxiv.org/abs/2502.13923) supply the model architecture.
  The exact downloaded [Qwen2.5-VL-3B-Instruct-AWQ checkpoint license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ/blob/main/LICENSE)
  is the **Qwen Research License Agreement**, rather than an assumed Apache license for
  model weights. The checkpoint is downloaded separately and is not redistributed here.
* [Hugging Face Transformers](https://github.com/huggingface/transformers) supplies the
  unmodified model modules (Apache-2.0).
* [PyTorch](https://github.com/pytorch/pytorch) supplies tensor operations and SDPA; see its
  [upstream license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
  FlashAttention is credited as prior work and was not evaluated in this Windows setup:
  [Dao et al.](https://arxiv.org/abs/2205.14135).

## Datasets and derived artifacts

* [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad), by
  Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger,
  *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and
  Localization*, IJCV 2022 ([DOI](https://doi.org/10.1007/s11263-022-01578-9)),
  is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  The full dataset and raw image directories are excluded. The selected montage
  `plots/edgeinspect/examples.png` contains resized source images with model-output annotations;
  its attribution, modifications, and ShareAlike terms are specified in
  [examples.LICENSE.md](plots/edgeinspect/examples.LICENSE.md).
* TextVQA: Singh et al., *Towards VQA Models That Can Read*, CVPR 2019
  ([dataset](https://textvqa.org/)). POPE: Li et al., *Evaluating Object Hallucination in
  Large Vision-Language Models*, EMNLP 2023 ([paper](https://arxiv.org/abs/2305.10355)).
  Preparation uses the `lmms-lab` Hugging Face releases; manifests and measured logs are
  retained, while source image downloads are excluded. Consult dataset source terms before use.
