# Contributing

## Set-up
```bash
conda env create -f environment.yml && conda activate SentAlign   # or any environment with torch + transformers
pip install -e ".[test]"        # compiles the Cython modules at install time
```
After changing a `.pyx` file run `pip install -e .` again (or `python setup.py build_ext --inplace`).

## Tests
```bash
python -m pytest                       # or: python tests/test_v1_characterization.py
```
* `tests/test_v1_characterization.py` replays the path search and post-processing of the 2023 code from saved
  embeddings and compares with `tests/golden/`: identical alignments, scores within 1e-6. Byte-identical output
  also depends on the BLAS library behind numpy (last digits of float32 scores), so it is checked only with
  `SENTALIGN_STRICT_BYTES=1`, on the machine that made the golden files. It needs no GPU and no model. It uses the small fixtures in
  `tests/fixtures/mini/`; with the full set (see below) it covers all 17 evaluation files.
* `tests/test_cli_basic.py`, `tests/test_import.py`: command line, exit codes, no side effects at import.
* `tests/test_anchors.py`: the anchor strategies on synthetic documents with known alignment (repeated sentences,
  short sentences, false matches in and out of order, abridged translations, no sure anchor), anchors from a file.
* `tests/test_galechurch.py`: Gale-Church beads equal those of the SentAlign 1.0 Cython code (golden files for the
  17 gold files; four small books through `SENTALIGN_BOOK_FIXTURES`), blank lines, the `galechurch` path finder.
* The Rust kernel (`rust/`, optional): `pip install ./rust` (or `maturin develop --release` in `rust/`). Its tests
  in `tests/test_pathfinding.py` are skipped when the package is not installed.
* `tests/test_pathfinding.py`: both kernels find the path of the 2023 code for every saved segment; the compiled
  kernel equals the Python reference bit for bit on random segments in both arithmetic modes; version-2
  post-processing (no sentence twice, order, the 2023 defects do not occur); slow: `--compat none` end to end.
* `tests/test_scoring.py`: spans and version-1 views against the saved overlap files and vectors, block scores
  against version-1 dot products, out-of-memory handling, batch order, cache, registry (fast, fake encoders); slow:
  LaBSE reproduces the saved vectors, and a second encoder runs end to end (downloads
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with `SENTALIGN_ALLOW_DOWNLOAD=1`).
* `tests/test_cli_e2e.py` (marker `gpu_or_cpu`, slow): runs the command on both evaluation sets and compares with the
  golden files. Needs the LaBSE model (`setu4993/LaBSE`) in the Hugging Face cache. Device: `cuda` if at least 6 GB
  are free, otherwise `cpu`; override with `SENTALIGN_DEVICE=cpu|cuda`. On `cuda` the output must be byte-identical,
  on `cpu` the alignments must be identical and the scores equal within 1e-5.
* Book fixtures (optional, large, outside the repository): `SENTALIGN_BOOK_FIXTURES=/path/to/book/fixtures`.

## Fixtures
The large fixtures (embeddings and stage results of all evaluation files, 40 MB) are not in git.
```bash
python tests/fetch_fixtures.py                    # download, verify sha256, extract to tests/fixtures/eval/
python tests/fetch_fixtures.py --from-dir DIR     # use a local copy of the archive instead
```
`tests/fixtures/mini/` (three smallest files per set) is committed and is used when the full set is absent.

## The compat rule
`--compat paper-2023` must keep reproducing the 2023 results: the characterisation tests and the end-to-end test
must stay green in that mode at every change. Fixes of known faults (marked `KNOWN FAULT` in the code) go
into new behaviour that is selected by the default preset, never into `paper-2023`.
