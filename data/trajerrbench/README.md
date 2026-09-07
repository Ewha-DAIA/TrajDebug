# TrajErrBench

TrajErrBench contains one-to-one Chinese (`zh`) and English (`en`) editions
for its 400-record τ²-Bench subset and 86-record SWE-Bench Pro subset. Every
JSON file uses TRAJDEBUG's unified trajectory schema and is ready for detector
input.

`manifest.jsonl` maps each language-neutral pair ID to its Chinese and English
files when an English pair is available. The paper reports results on the
Chinese edition.

The English files are constructed from aligned source trajectories while
inheriting the Chinese release's task ID, reward, and critical-error labels.
Metadata is rebuilt from a strict allowlist:

```bash
python -m data_processing.build_trajerrbench_english \
  --tau2bench-failed-trajectories /path/to/tau2/failed_traj \
  --swebenchpro-processed-en /path/to/Swe-Bench-Pro/processed_en
```
