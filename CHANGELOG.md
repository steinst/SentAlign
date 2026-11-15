# Changelog

## 2.0.0 (2026-11-15)

SentAlign 2.0, the first release on PyPI (`pip install sentalign`, optionally `sentalign[rust]`). With
`--compat paper-2023` the alignments are those of SentAlign 1.0 and the 2023 paper (tag `v1.0-paper`); the default is
version 2, with the faults of version 1 fixed. The parts below are in the order they were made, the latest first.

### Release

* Version `2.0.0` of `sentalign` and `sentalign-rust`; development status "Production/Stable", classifiers for
  Linux, macOS and Windows and Python 3.10 to 3.14.
* README: starts with what a sentence aligner does and the problem it solves, with an animated figure of SentAlign's
  steps on English-Icelandic subtitles from the evaluation set (`docs/images/how-it-works.gif`), then what is new
  in 2.0, installation and usage.
* A pushed tag `v<version>` uploads both packages to PyPI (trusted publishing; the workflow checks that the tag names
  the version of both) and then installs `sentalign[rust]` from PyPI on every system and aligns a text with it.

### The CPU is used when there is no GPU

* `--proc-device` has a new default, `auto`: `cuda` when PyTorch finds a CUDA device, otherwise `cpu` (with a
  warning that this is much slower). Before, the default was `cuda`, and on a machine without CUDA the `sentalign`
  command failed every file unless `-device cpu` was given. The Python API already chose the device this way.
  The device is logged and recorded in the `.json` file; `--half` is on by default only when the device used is
  `cuda`. Output on a GPU is unchanged, also with `--compat paper-2023`.
* `--proc-device cuda` (or `device='cuda'` in Python) on a machine without CUDA is now refused at once (exit code 2;
  `ValueError` in Python) instead of failing when the model is loaded.
* README: the numbers that came from earlier development runs were measured again with this version and all hold
  (`--reward squared-cosine`, `--length-prior`, `--pathfinder galechurch`, the four `chain` configurations, the speed
  of `--half`, the cutoffs of MiniLM and mpnet). Changed: the Rust kernel on the 22,000 x 24,000-sentence book takes
  20 s against 31 s solved whole and 3.1 s against 3.5 s with anchors (was 19 / 30 and 3.0 / 3.5), and the best cutoff
  of `laser2` is 0.4-0.5 (was 0.4-0.55).
* CI: the end-to-end job may take 300 minutes (was 120, which was not enough for the slow tests on the Linux and
  Windows runners); the wheel check runs the version 2 alignment without `--proc-device`.

### README and citation

* README rewritten for the release: installation from PyPI (with the PyTorch notes for Windows GPUs, machines
  without a GPU and Intel Macs), both ways of running, the output files, exit codes, the Python API, one table of all
  options (with both defaults where `--compat paper-2023` differs) and tables of the `vecalign` and `chain` settings.
* Reproducing the paper: the commands for both evaluation sets with `--compat paper-2023` and
  `evaluate.py --lax-recall paper-2023`, and those of version 2, with the numbers of new runs. The README of the
  development versions gave the paper's German-French numbers (strict F1 0.932) next to the version 2 command; that
  command gives 0.931 (strict P / R 0.934 / 0.928). English-Icelandic: 0.951 in both modes.
* Publications: every entry that SentAlign can print, with its BibTeX and when to cite it. The BibTeX of the 2023
  paper now has its pages, DOI and URL (in `--bibtex` output too).
* New `CITATION.cff` (GitHub's "Cite this repository"; the preferred citation is the 2023 paper). Licence line:
  Copyright 2023-2026.

### Wheels for more systems, test release

* A release candidate, `2.0.0rc1` of both packages, was uploaded to TestPyPI by a manual run of the workflow.
* Wheels for Python 3.10 to 3.14 (3.14 is new) on Linux x86_64 and aarch64 (new), macOS arm64 and x86_64 (new) and
  Windows x86_64; `sentalign-rust` wheels for the same systems and a source package of it.
* Each wheel is checked in CI on its own system: installed with its dependencies (CPU PyTorch) in a fresh Python, it
  aligns a text of the German-French evaluation set with the `sentalign` command, and the output is compared with
  that of SentAlign 1.0. The source package is installed once to check that the C extensions compile from it.
* Intel Macs: PyTorch 2.2 is the last release for them, so there `sentalign` asks for `numpy<2` and
  `transformers<5` (transformers 5 needs a newer PyTorch). Python 3.10 to 3.12 only, since PyTorch 2.2 has no wheels
  for later versions.

### Fixes (output files, options, annotation tool)

* `--pathfinder vecalign`, `chain` and `galechurch`: a link with more words than `-maxwords` (or fewer than
  `-minwords`) is written with its score again. Since the word-limit fix below it came out with score 0 (for example
  a 131-word pair, 0.99998575 before), although these path finders do not apply the word limits. The default path
  finder is not affected.
* A file that fails no longer takes the output of another file with it: for a failed `news`, the cleanup also removed
  `news.2020.txt.path`, `.aligned` and `.json` (any name beginning with `news.`). Only the temporary files of the
  failed file are removed now.
* When a group of small files (`--batch-texts`) cannot be encoded together, the files are encoded one by one. Before,
  the first file of the group was reported as failed without being tried. `SentAlign.align_many` likewise.
* Two instances that reach the same file: the temporary output names now carry the process id, and a finished
  `.aligned` is only removed first with `--overwrite` (or when it has no `.path`). Before, the second instance could
  delete the finished `.aligned` of the first and then fail.
* `sentalign SRC TRG -o OUT`: the languages are what `-sl` / `-tl` say. Without them the run record (`.json`) has no
  language (it said `eng` / `isl` for any text), and an encoder that needs the language (LASER) is refused with a
  usage error (before: a warning, and the `eng` / `isl` encoders). An explicit `-sl eng -tl isl` no longer warns.
  The `-dir` form keeps the defaults `eng` / `isl`.
* `--max-concatenations` above 15 with `--compat paper-2023`: the message now names `--path-backend legacy`, the 2023
  path search, which has no such limit.
* Annotation tool: leaving the page while Finish is being saved no longer sets the window back to draft (a draft
  without any change after Finish is not sent). A change made after Finish is still saved as a draft.
* Texts: `--help` of `--max-segment-nodes` and `--max-concatenations` (limit of 15). `--help`, README and this
  file named options that do not exist; they now say `--proc-device` and `-minwords` / `-maxwords`.

### Fixes (batching)

* `--batch-texts`: the groups of small files are now encoded one at a time, each just before its files are aligned,
  and the vectors of a file are released when it is done. Before, every group of the run was encoded before the
  first file was aligned and all vectors were kept (about 3 KB per text, so a large corpus of small files could
  run out of memory). Same groups, same output; `SentAlign.align_many` likewise.
* A scorer of your own (`--scorer`) that has only `prepare` and `block` no longer fails with `AttributeError:
  ... 'encode_sides'` when several files (or `align_many` pairs) are aligned with `--compat none`. Batching is an
  optional capability of a scorer (`encode_sides`, `use_vectors`; see `docs/extending.md`); without it every
  document is prepared on its own.

### Fixes (annotation tool)

* Finish is no longer undone by an autosave: the pending draft is cancelled when any save starts, and saves are sent
  one after another, so a late draft cannot overwrite `done`.
* Finishing a window checks the sentence numbers: integers, inside the document on both sides, in the links and in
  the lists of unaligned sentences. Before, a number outside the document (or not a number) was accepted.

### Fixes (CLI, API, output)

* `sentalign SRC TRG -o OUT` keeps the languages of `-sl` / `-tl` (it set them to the names of its temporary folders,
  `src` / `trg`, which LASER could not use). Without `-sl` / `-tl` the defaults `eng` / `isl` are used, and a warning
  says so when the encoder needs the language (LASER).
* `sentalign.align(..., pathfinder_options={...})` no longer fails (the options dictionary could not be a cache key).
* `--overwrite` writes the new `.path` and `.aligned` under temporary names and renames them last, after removing the
  old `.aligned`. A failure part-way no longer leaves a new `.path` with the old pairs counted as done.
* `--max-segment-nodes` is also checked for a document that is not cut (`--anchors none`, or below the anchoring
  limit), for the path finders that use anchors (not Vecalign and chain, which take whole documents).
  `--compat paper-2023` is unchanged.

### Fixes (aligner)

* A document or segment of exactly 2 sentences no longer crashes the path search (`ValueError`, broadcasting); this
  hit all four path finders, and 2-sentence segments made by `--anchors file:`.
* `--max-concatenations` above 15 is refused with a clear message (the back-pointers of the kernels are one byte; at
  16 a 16:16 link silently became "skip target"). The default (4) is unchanged.
* Post-processing (re-evaluation, absorption) keeps the word limits (`-minwords`, `-maxwords`) of the path search, so it can
  no longer build a link the search would refuse. `--compat paper-2023` is unchanged.
* A real sentence that contains the word `NULLALIGN` is no longer dropped from the `.aligned` file (only rows with an
  empty side are). `--compat paper-2023` keeps the 1.0 behaviour.

### Evaluation scripts

* `evaluation/evaluate.py`, `evaluate_windows.py`: lax recall is now the lax recall of Bleualign and Vecalign (a gold
  alignment is found when a system alignment overlaps it on both sides). The 1.0 script divided the lax true
  positives of the system by those plus the gold alignments without an exact match, so lax recall and lax F1 came
  out lower: German-French 0.933 / 0.964 before, 0.980 / 0.989 now; English-Icelandic 0.955 / 0.970 before,
  0.988 / 0.986 now (SentAlign 1.0 output). `--lax-recall paper-2023` (`lax_recall='paper-2023'`) gives the numbers
  of the 2023 paper. Strict scores and lax precision are unchanged.
* No error when a system output or a gold file has no alignments left to score (all scores 0).

### Encoders with their own code

* `--trust-remote-code` (off by default) lets `--encoder` load a model that ships its own code.

### A second link reward

* `--reward squared-cosine` (`reward='squared-cosine'`): the link weight of ChainAlign (score squared, free skips,
  default cutoff 0.3) in the SentAlign search; all kernels, same output. `--length-prior` adds ChainAlign's
  length-ratio prior. The default is unchanged (`--reward sentalign`, cutoff 0.4).

### `chain` path finder

* `--pathfinder chain`: seed, chain, extend, re-implemented after ChainAlign (Goldner and Gramazio, WMT 2026) from
  the paper's algorithms. German-French evaluation set: strict F1 0.941 (paper 0.941), and the three other rows of
  the paper's ablation table are reproduced too; English-Icelandic 0.946.

### Vecalign as a path finder

* `--pathfinder vecalign`: Vecalign's search code is included in the package (taken from the Vecalign repository,
  Apache-2.0; compiled extension with a Python fallback) and runs on the scorer's vectors; same beads as the
  `vecalign` command on the same vectors (17 of 17 evaluation files).
* Interface: a segment can hand out span vectors (`Segment.src_vectors(k)`, `trg_vectors(k)`, `text_vector`;
  `Scorer.span_vectors`, `Scorer.text_vector`); path finders have their own options (`option_defaults`,
  `--pathfinder-opt NAME=VALUE`) and can decline anchors (`uses_anchors = False`).

### Encoders by name, LASER, pre-filled annotation

* `--encoder labse|bge-m3|minilm|mpnet|laser2|laser3`: the name sets the encoder kind and pooling.
* LASER encoders (`sentalign.scoring.laser`, optional package `laser_encoders`); encoders with
  `needs_language = True` are given the language of each side (`Side.lang`).
* Annotation tool: `prepare.py prefill` aligns the pairs with SentAlign; the annotator corrects the pre-filled windows.

### `--half` is the default on a GPU

* With `--proc-device cuda` and `--compat none` the encoder now runs in half precision unless `--no-half` is given
  (`half=False` in Python). `--compat paper-2023` and the CPU stay in float32. Results: see below.

### `--half`

* `--half` (`half=True` in Python): the encoder runs in float16 on the GPU, vectors are returned as float32. On the
  evaluation sets the alignments are identical to float32 (both sets, all 17 files); on two long books 1 and 2 links
  of 6,400 and 17,700 differ. Encoding is 2.8 to 3.1 times faster (6,900-sentence book 69 s to 25 s, 19,500-sentence
  book 156 s to 50 s). Explicit `--half` is refused on the CPU and reported as a deviation with `--compat paper-2023`.

### Rust kernel, faster score blocks

* Optional package `sentalign-rust` (`rust/`, PyO3 + maturin): the path-search kernel in Rust,
  `--path-backend rust`. Bit-for-bit the same back-pointers and sums as the Python reference and the Cython kernel,
  in both arithmetic modes. With `-proc N` it computes the tiles of each anti-diagonal on N cores.
* Measured (47 million nodes, kernel only): Cython 26 M nodes/s, Rust on one core 25 M nodes/s (paper arithmetic 19
  against 21), Rust on 16 cores 110 M nodes/s.
* Score blocks: the blank-line masks are computed once per segment instead of on every block, and the scorer writes
  straight into the band (no extra copy). Path search of the largest test book solved whole: 40 s to 30 s with the
  Cython kernel, 19 s with the Rust kernel on 16 cores. Outputs unchanged.
* CI builds the Rust kernel on Linux, Windows and macOS and runs its equality tests; `wheels.yml` builds its wheels.

### Python API, many files, documentation

* **Python API** (`sentalign.api`): `sentalign.SentAlign(**options)` loads the model once; `.align(src, trg)` and
  `.align_many(pairs)` return an `Alignment` (links with sentence numbers and scores, `aligned()`, `pairs()`,
  `to_path_lines()`, `to_dict()`); `sentalign.align(src, trg, **options)` uses a shared aligner. `import sentalign`
  still does not import torch.
* **Small files are encoded together** (`--batch-texts N`, default 200,000 texts per group with `--compat none`):
  few full encoder batches instead of many nearly empty ones, and texts shared by several files are encoded once.
  Same alignments as one file at a time; for 10 distinct ParIce files the encoding is about 20 % faster, for a
  corpus with repeated passages much more (120 files made from 10 repeated 12 times: 77 s to 9 s). Not used in
  paper-2023, where the batch composition would change the last digits of the vectors.
* The extensions are optional at installation: without a C compiler SentAlign installs and falls back to the
  readable Python path search (slow); `--anchors legacy-greedy` then reports that the module is missing.
* A scorer can say it needs no encoder (`needs_encoder = False`): no model is loaded.
* Documentation: how it works, `docs/extending.md` with a working example for a scorer, a path finder and an anchor
  strategy (tested), what `--compat paper-2023` guarantees.
* Wheel and source-distribution workflow (`wheels.yml`, manual or on a version tag; nothing is published).

### Anchoring; version 2 becomes the default

* **`--compat none` is now the default.** `--compat paper-2023` reproduces SentAlign 1.0.
* Anchoring is a separate part (`sentalign.anchors`): `AnchorStrategy`, `--anchors NAME`, entry-point group
  `sentalign.anchorers`. Strategies: `mutual-best` (default), `gale-church`, `file:PATH`, `none`, `legacy-greedy`.
* The node limit before anchoring is 100 million by default (4 million in paper-2023): with the compiled path search
  a book of 10,000 x 10,000 sentences is solved whole, without anchors.
* `mutual-best` accepts a pair only if each sentence is the other's best match in the document, 0.05 ahead of the
  second best on both sides, both at least 4 words; candidates out of order with the rest (longest increasing chain)
  or without neighbours are dropped; as few anchors as needed are used; a segment without a sure anchor is solved
  whole (refused above `--max-segment-nodes`).
* Removed: the mid-size anchoring of version 1 and its two Cython modules (`anchoring.pyx`, `utilities.pyx`). It
  could not work (TypeError on every call, an out-of-bounds read that crashed the process, an integer fallback
  matrix) and ended in a crash or in solving the whole graph. paper-2023 mode now solves such files
  whole, which is what version 1 did when it did not crash. This removes the segmentation fault.
* The large-file anchoring of version 1 is kept bug for bug as `legacy-greedy` (`sentalign.legacy.greedy_anchors`,
  `legacy/greedy.pyx`); paper-2023 uses it above `-maxgc` sentences. Output byte-identical to version 1 on two books.
* The aligner module shrank from 837 to about 480 lines; the JSON record has an `anchoring` section.

### Gale-Church from the galechurch package

* The bundled Cython Gale-Church code and its backward file reader (`galechurch.pyx`, `file_read_back.pyx`,
  `buffer_work_space.pyx`, about 540 lines) are gone; `sentalign.gale_church` uses the `galechurch` package
  (now a dependency). Same beads as the old code on all 17 gold files and 9 books (the largest 6,900 sentences),
  25-30 times faster (6,900 x 6,900 sentences 3.7 s, was 99 s); a 22,000 x 24,000 book takes 41 s and 0.5 GB.
* Blank lines : the old code cut the file at blank lines and numbered each block from 0, so
  anchors pointed at the wrong sentences. Passing a blank line as a zero-length sentence is no solution either:
  Gale-Church gives two empty sides the cost minus infinity (the original program and the package do), which
  turns the alignment into nonsense. Blank lines are now left out and the beads keep the numbers of the file.
* `--pathfinder galechurch` (with `--compat none`): Gale-Church alone as a baseline aligner; links are scored with
  the scorer, no post-processing. `PathFinder.postprocess` tells the pipeline whether to post-process.

### Path search and post-processing

* The path search is a separate part (`sentalign.pathfinding`): a `PathFinder` gets a segment (scores by blocks, word
  counts) and returns the path; `--pathfinder NAME`, entry-point group `sentalign.pathfinders`. `sentalign-dp` is the
  algorithm of the paper, with two kernels that give identical results: a compiled one (`_dp_cython.pyx`) and a
  readable reference (`_dp_python.py`, `--path-backend python`). The kernel works on score blocks in bands of rows,
  keeps one byte per node, and runs segments in threads (nothing is copied to worker processes any more).
* `--compat paper-2023` (still the default): the new kernel with the arithmetic and the embedding lookups of the 2023
  code, followed by the 2023 post-processing. Output byte-identical to SentAlign 1.0 on the gold sets and on six
  books; `--path-backend legacy` runs the 2023 path search itself.
* `--compat none`: double-precision path search and a new post-processing (`sentalign.postprocess.reevaluate`):
  links only of consecutive sentences, no sentence in two links, unaligned neighbours absorbed best gain first,
  result checked. Blank lines are never linked. No duplicate rows for unaligned sentences.
* Speed: path search and post-processing of a 1,900-sentence book 0.3 s (2023 code: about 2 minutes); a
  6,900-sentence book without anchors (47 million nodes) 4 s.

### Scoring

* The scoring is a separate part (`sentalign.scoring`): a `Scorer` answers similarity for whole blocks of sentence
  spans (`block`, `pair`); the default `EmbeddingScorer` uses an `Encoder`. `HFEncoder` takes any Hugging Face model
  id or folder (`--encoder`), with `--pooling pooler|cls|mean`, `--model-revision`, `--max-tokens`. New scorers and
  encoders can be registered (entry-point groups `sentalign.scorers`, `sentalign.encoders`).
* Spans and vectors live in arrays (`sentalign.document.Side`); the temporary overlap, `.used`, `.lengthok`,
  `.linenumbers` and `.labse_emb` files are gone. The version-1 path search and anchoring get their dictionaries and
  lists from in-memory views with the same content.
* Out of GPU memory while encoding: the batch is halved and the work continues; at batch size 1 a clear error.
  Version 1 retried the same call and appended partial results, so vectors could be paired with the wrong texts
  or it saved nothing and crashed later.
* `--compat none` encodes in length-sorted batches and identical texts once (same alignments on the gold sets,
  about 30 % faster on the German-French set); `--compat paper-2023` (default) encodes exactly as version 1.
* Optional embedding cache (`--embedding-cache DIR`). The JSON run record has the encoder, its resolved revision
  and any setting that makes a paper-2023 run differ from the paper.
* `sentalign.embeddings` moved to `sentalign.legacy.overlap_files` (reference implementation for the tests).

### Package, no change of behaviour

This part only restructures the code; the alignments are those of the 2023 paper (`--compat paper-2023`, at this
point the only behaviour) and are byte-identical to the golden files of tag `v1.0-paper` on the GPU.

* Installable package `sentalign` (`src/` layout, `pyproject.toml`); the Cython modules are compiled at install time,
  nothing is compiled or written at import time (no `pyximport`, no `.so.reload1` files).
* `AlignConfig` dataclass and an `Aligner` class replace the module-level globals and argument parsing of `sentAlign.py`.
* New command `sentalign` (also `python -m sentalign`): the old flags plus `SRC_FILE TRG_FILE -o OUT_DIR`,
  `--list-files`, `--overwrite`, `--quiet`, `--verbose`, `--compat`, `--version`.
* No more `files2align.txt` / `filesdone.txt`: files are processed in sorted order, a file is done when `.path` and
  `.aligned` exist, outputs are written to a temporary name and renamed. `sentalign-files` keeps the old lists.
* `--filename` works, `-proc` is an integer, the temporary folder is removed with `shutil.rmtree`.
* `logging` with timestamps and stage durations; a JSON run record `<name>.json` next to each output; the `.info` file
  is still written. Exit code 0 / 1 (a file failed) / 2 (usage error); a failing file is not marked done.
* Bare `except:` replaced by named exceptions; swallowed errors are logged once per place and file.
* Removed `repeatedTimer.py`, `reportInfo.py`; `sentence_split.py` and `generate_aligned_lines_with_nulls.py` moved to `scripts/`.
* Tests: characterisation tests run under pytest and as a script, CLI tests, end-to-end test against the golden
  files, fixture download script.
* Known faults of version 1 are kept on purpose and marked `KNOWN FAULT` in the code; they are fixed in version 2 (the default).
