# EdgeInspect-VLM — interview preparation (measured facts only; final numbers filled from the full grid)

## Why MVTec LOCO AD
It is the standard industrial anomaly benchmark that contains **both** structural defects
(scratches, contamination, deformation) and **logical** defects (a missing/extra/misplaced
component, a wrong count or arrangement) with original train/validation/test splits and only
*normal* training images — exactly the few-shot, normal-reference setting of real inspection
lines. Logical anomalies are locally normal, so they test whether a VLM actually *compares*
the query against the references rather than looking for local damage.

## Structural vs logical anomalies
Structural anomalies are visible as local texture/shape changes (a crack in a pushpin, a stain
on a juice label). Logical anomalies violate a global constraint (two pushpins in one
compartment, a missing tangerine in the breakfast box, a cable connecting the wrong terminal
counts); each part looks fine, only the composition is wrong. Patch-level detectors handle the
first well and the second poorly; the hope for VLMs is global reasoning over the references.

## Why quantization alone was insufficient
INT4 AWQ shrinks the resident weights to ≈3.3 GB, but the *prompt* grows with every image:
≈493 visual tokens per 512-token-budget image, i.e. 4,437 visual tokens at k = 8. Measured on
the RTX 4060 (isolated runs, pushpins, SDPA, tuned AWQ dispatch): at 100% tokens the device
footprint is 4.9 GB (k=1), 6.1 GB (k=4) and 8.6 GB at k=8 — over the 8 GB card once the
≈1.1 GB CUDA context and other GPU processes are counted, and at k=12 it reaches 12.3 GB, spills
to system RAM and a single query takes 104 s instead of ~6 s. Quantization fixes the weights;
it does nothing for activation/KV memory and prefill time, which scale with visual tokens.

## Why visual-token compression matters for multi-image inference
The same retention ratio removes (k+1)x more tokens in a k-shot prompt, and memory/prefill are
dominated by those tokens: at 50% retention the largest k that stays within VRAM rises from 4
(100%) to 12 (7.6 GB), and at 75% from 4 to 8 (7.4 GB, 4.9 s/query). Unlike the single-image
case (EdgeCompose), where the vision encoder set the memory peak and compression saved < 2%,
multi-image prompts make the LLM prefill the memory peak, so compression directly buys
references.

## Failed experiment
The first classification prompt produced "ANOMALOUS" for 100% of queries (normal ones
included), so raw-answer F1 equals the trivial always-anomalous baseline. Instead of tuning the
prompt on test anomalies (which would leak labels), the decision uses exact answer likelihoods
with a threshold calibrated on *normal validation images only*.

## Engineering surprise
On Windows, exceeding VRAM does not raise CUDA OOM: the driver's sysmem fallback silently pages
to system RAM, so "it fits" must be verified by footprint and latency, not by the absence of an
exception.
