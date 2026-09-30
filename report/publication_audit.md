# Publication audit

Prepared on 1 October 2026. No research experiments were rerun.

## Provenance and Git dates

The owner requested one daily contribution from **31 July through 30 September 2026**.
The existing research-checkpoint changes were split into **62 nonempty commits**, one per
day. Every commit uses G Abhinav Reddy <abhinavreddyg08@gmail.com> as author and committer,
and both Git timestamps are assigned to that day's date. These are owner-requested
publication dates, not evidence that experiments ran on those days. Experimental timestamps
remain intact; the original eight research commits were recorded on 30 September–1 October.
The original release-preparation timestamp was 1 October 2026 at 12:18:28 +0530.

Each research checkpoint's tree is reconstructed exactly from the sanitized source history.
The daily split preserved the final source, configs, tests, measured logs, aggregates,
manifests, and research plots. Subsequent publication presentation changes are limited
to documentation and the architecture illustration, as listed below.
Original histories remain in local backup branches and a verified Git bundle, which are not
pushed. Publication history corrections update main and the release tag atomically with explicit
force-with-lease checks against reviewed remote refs; they abort on intervening remote work.

| Original research checkpoint | Last daily commit for checkpoint | Date |
|---|---|---|
| `88c1af6d7a3339ecf05ac261c935be89813f0b31` | `b4cc9ebe47215f4790a6415f91965618f6428b38` | 2026-08-12 |
| `62b8e0842f718cd7a63efab51becad4eb26aaea1` | `12a5f05bdee81f5fdc7f4f290266de5fda39534c` | 2026-08-26 |
| `2c763098fff32dec360961de0b1236d31def6fc2` | `322ac65dcb3bffabca2600c22507e0724353f952` | 2026-09-11 |
| `5f1a6d62dd863affaf3e1b49bfb10ce498601acb` | `775b1d2db32598b442af9f6b9c130b1bd10f27f4` | 2026-09-16 |
| `0f0876337df3e0c70e0ad58aca363bd4e34ea0b1` | `e2e6f0d35ae448f6cf3841ad8dedf2e7d1b8c72b` | 2026-09-20 |
| `39ee66e546210bd985df76aaa13df1c6f9cf7369` | `e4282839fa80a0aeaac696d81b7c7469a8c75173` | 2026-09-24 |
| `dd0a0a10583e7ac162b05b36e917cf6f63e8acc4` | `35bfbd36c9f173a4fa3b18b6f6bb55dd441d79af` | 2026-09-25 |
| `471921a5293678093b89617d96ad69535e315af9` | `418cc07de4dfcecfe4a789f7daf0dbd1c00c2620` | 2026-09-29 |

The final publication-preparation commit is assigned to 30 September 2026.

| Daily publication date | Commit purpose |
|---|---|
| 2026-07-31 | Add environment and shared utilities; Qwen inference; attention backends; visual token compression |
| 2026-08-01 | Add latency, memory and power profiling; VQA data loaders |
| 2026-08-02 | Add VQA scoring; benchmark engine; VQA analysis |
| 2026-08-03 | Add deployment selection; smoke and diagnostic tools |
| 2026-08-04 | Add inference configs: SDPA; inference configs: eager |
| 2026-08-05 | Add inference configs: fa2; inference configs: sweep; inference configs: uniform |
| 2026-08-06 | Add unit tests |
| 2026-08-07 | Add evaluation manifests: POPE; evaluation manifests: TextVQA; evaluation manifests: provenance |
| 2026-08-08 | Add VQA records: POPE, SDPA; VQA records: TextVQA, SDPA |
| 2026-08-09 | Add VQA records: TextVQA, eager; VQA records: TextVQA, uniform |
| 2026-08-10 | Add VQA results: aggregate and Pareto tables; VQA results: paired/query tables; saved environment, smoke and demo output |
| 2026-08-11 | Add VQA plots |
| 2026-08-12 | Add project status and run provenance |
| 2026-08-13 | Update environment and shared utilities; benchmark engine; AWQ dispatch; VQA analysis; smoke and diagnostic tools |
| 2026-08-14 | Update inference configs: sweep; inference configs: tuned; unit tests |
| 2026-08-15 | Update VQA records: POPE, SDPA; VQA records: POPE, dispatch checks |
| 2026-08-16 | Update VQA records: POPE, eager; VQA records: POPE, repeatability |
| 2026-08-17 | Update VQA records: POPE, uniform; VQA records: TextVQA, SDPA; VQA records: TextVQA, decode profiling |
| 2026-08-18 | Update VQA records: TextVQA, dispatch checks |
| 2026-08-19 | Update VQA records: TextVQA, eager |
| 2026-08-20 | Update VQA records: TextVQA, repeatability; VQA records: TextVQA, uniform |
| 2026-08-21 | Update VQA results: aggregate and Pareto tables |
| 2026-08-22 | Update VQA results: paired/query tables; VQA results: precision/quality; VQA results: system profiling; saved environment, smoke and demo... |
| 2026-08-23 | Update VQA plots |
| 2026-08-24 | Update exploratory notebook |
| 2026-08-25 | Update report: report methods |
| 2026-08-26 | Update README results and reproduction; project status and run provenance |
| 2026-08-27 | Update Qwen inference; visual token compression; VQA analysis; deployment selection |
| 2026-08-28 | Update LOCO data loading; reference sampling; inspection prompts |
| 2026-08-29 | Update inspection classification and explanations; anomaly metrics; few-shot analysis; inspection runner |
| 2026-08-30 | Update inspection memory probing; inspection examples and demo; unit tests |
| 2026-08-31 | Update inspection manifests: breakfast_box |
| 2026-09-01 | Update inspection manifests: juice_bottle |
| 2026-09-02 | Update inspection manifests: pushpins |
| 2026-09-03 | Update inspection manifests: screw_bag |
| 2026-09-04 | Update inspection manifests: splicing_connectors |
| 2026-09-05 | Update VQA records: POPE, SDPA; VQA records: POPE, eager; VQA records: POPE, tuned; VQA records: POPE, uniform |
| 2026-09-06 | Update VQA records: TextVQA, SDPA; VQA records: TextVQA, eager |
| 2026-09-07 | Update VQA records: TextVQA, tuned; VQA records: TextVQA, uniform; inspection records: development |
| 2026-09-08 | Update VQA results: aggregate and Pareto tables; VQA results: paired/query tables; VQA results: precision/quality; saved environment, smo... |
| 2026-09-09 | Update VQA plots |
| 2026-09-10 | Update report: edgecompose report; report: edgeinspect intro; report: report; report: report methods |
| 2026-09-11 | Update README results and reproduction; project status and run provenance |
| 2026-09-12 | Update Qwen inference; benchmark engine; smoke and diagnostic tools; few-shot analysis; inspection runner; inspection memory probing; ins... |
| 2026-09-13 | Update VQA records: TextVQA, precision validation; inspection records: development |
| 2026-09-14 | Update VQA results: precision/quality; inspection results: per-query/category tables; inspection results: seed variability; inspection re... |
| 2026-09-15 | Update inspection plots: development |
| 2026-09-16 | Update report: edgecompose career; report: edgecompose report; README results and reproduction; project status and run provenance |
| 2026-09-17 | Update VQA analysis; few-shot analysis; inspection runner; inspection records: allocator archive; inspection records: seed-0 |
| 2026-09-18 | Update inspection results: per-query/category tables; inspection results: reference capacity; inspection results: seed variability; inspe... |
| 2026-09-19 | Update inspection plots: development; inspection plots: grid |
| 2026-09-20 | Update report: edgeinspect interview draft; report: edgeinspect setup; README results and reproduction |
| 2026-09-21 | Update deployment selection; anomaly metrics; few-shot analysis; inspection runner; inspection examples and demo; unit tests |
| 2026-09-22 | Update inspection records: seed-0; inspection records: seed-0 archive; inspection records: seed-1 |
| 2026-09-23 | Update inspection results: paired contrasts; inspection results: per-query/category tables; inspection results: reference capacity; inspe... |
| 2026-09-24 | Update inspection plots: final figures; report: edgeinspect interview draft; report: edgeinspect intro; report: edgeinspect report; repor... |
| 2026-09-25 | Update inspection records: seed-1; project status and run provenance |
| 2026-09-26 | Update few-shot analysis; inspection records: seed-1 |
| 2026-09-27 | Update inspection results: paired contrasts; inspection results: per-query/category tables; inspection results: reference capacity; inspe... |
| 2026-09-28 | Update inspection plots: final figures; inspection plots: qualitative examples |
| 2026-09-29 | Update report: edgeinspect report; report: interview prep; report: merl application; README results and reproduction; project status and ... |
| 2026-09-30 | Prepare public GitHub release and daily commit mapping |

## Publication changes

* Expanded `.gitignore` for credentials, environments, caches, weights, archives, raw
  datasets, and temporary output; final manifests, results, reports, and figures remain tracked.
* Sanitized machine paths in `PROJECT_STATUS.md`, the environment snapshots,
  `results/system_info.json`, and `results/edgeinspect/loco_statistics.json`.
  Historical versions of those five provenance files were sanitized as well.
  Local package build URLs were replaced with the installed distribution versions.
* System-info collection now records the interpreter filename instead of its absolute path.
* README setup now specifies an isolated Python 3.9 environment, pinned NumPy/Hub versions,
  the archive location, PowerShell-compatible commands, a runnable demo path, compatibility
  workarounds, and the distinction between saved fp16 measurements and current bf16 defaults.
* Standardized the inspection contrast to +0.093 at three decimals in the README,
  EdgeInspect report, one-page summary, and interview material. The aggregate's unrounded
  value remains 0.09250000000000003 and its CI remains [0.026475, 0.1695125].
* Clarified two report phrases as 75% token **retention**, rather than ambiguous pruning wording.
* Added adapted-code copyright/change notices, upstream Apache-2.0 and AutoAWQ MIT license
  copies, `THIRD_PARTY_NOTICES.md`, and a CC BY-NC-SA 4.0 attribution/license sidecar for
  the selected dataset-image montage. Corrected statements implying no dataset images appear.
* Added this release audit and commit mapping. No overall project license was invented.
* Enlarged the README architecture into an accessible full-width SVG with separate VQA
  and inspection paths, shared inference stages, profiling, analysis, and deployment selection.
  The editable Mermaid version remains available below the illustration.

* Removed the private Conda environment identifier from every lockfile version in the
  published branch/tag history, preserving all package pins and daily Git timestamps.

## Local verification

* 44 unit tests passed.
* Manual redacted pattern scan of all Git blobs and tracked files found no secret candidates.
  Neither gitleaks nor trufflehog was installed; this is not a claim of formal exhaustive detection.
* No weights, raw dataset directories, local environments, or downloaded archives are tracked.
* Largest tracked/historical blob: `results/aggregate/per_query_500.csv`, 8,721,109 bytes.
  No blob exceeds 10 MB, 50 MB, or 100 MB.
* 74 Python modules parsed; 82 JSON/notebook files, 112 JSONL files, 38 CSV files,
  21 YAML configs, and 40 PNG figures validated. All five archived seed-0 checksums match.
* All 17 scripts with argparse entry points returned successful `--help`; internal
  diagnostic/example scripts without parsers were inspected without executing inference.
* Final inspection logs contain 5,500 rows and zero failures, including calibration rows.
* Relative aggregates confirm 27.8015% TextVQA latency reduction, 99.6385% quality retention,
  26.8112% GPU-energy reduction, and 38.6963% POPE latency reduction. SDPA reference capacity
  is 4/8/12/16 at 100/75/50/25% retention.
* Relative Markdown links in the README and reports resolve. Feasible/infeasible optimizer
  output is checked from the saved results, without loading the model.

## Visitor caveats

The project has no overall license. Adapted code, the selected CC BY-NC-SA dataset montage,
and separately downloaded Qwen Research License checkpoint have distinct terms; consult
`THIRD_PARTY_NOTICES.md`. The full model and datasets must be obtained separately.
Results are hardware/environment-specific. Original fp16 sweep measurements and the bf16
erratum remain documented. Inspection performance remains modest, pushpins is near chance,
generated labels are heavily biased, and AUROC >= 0.80 is infeasible in the measured grid.
