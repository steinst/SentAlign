# SentAlign

SentAlign is a sentence alignment tool for parallel corpora. 
It uses LaBSE embeddings to find sentence pairs that are similar in meaning 
and an alignment algorithm based on Dijkstra's algorithm to find the optimal alignment.

### License 
Copyright 2023 Steinþór Steingrímsson
SentAlign is released under the [Apache License, Version 2.0](LICENSE).


### Building the environment
The environment can be built using the provided environment.yml file:
```bash
conda env create -f environment.yml
```

### Running the alignment
We assume that the documents to be aligned have the same names in the source and target language, but are kept in folders named using the language code. For example, if we want to align the files in the folder `/path/to/files` we would have the following structure:
```bash
/path/to/files/eng/file1.txt
/path/to/files/eng/file2.txt
...
/path/to/files/isl/file1.txt
/path/to/files/isl/file2.txt
```

Assuming a Conda environment has been built as described above, the environment has to be activated before SentAlign is run:
```bash
conda activate SentAlign
```

Start by creating a list of files to align:

```bash
python3 files2align.py -dir /path/to/files --source-language eng
```

Then you run the alignments. Aligning English and Icelandic files:

```bash
python3 sentAlign.py -dir /path/to/files/to/align -sl eng -tl isl
```

## Citation

```bibtex	
@inproceedings{sentalign-2023,
    title = "SentAlign: Accurate and Scalable Sentence Alignment",
    author = "Steingrímsson, Steinþór and
      Loftsson, Hrafn  and
      Way, Andy",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
}
```
