#SentAlign

SentAlign is a sentence alignment tool for parallel corpora. 
It uses LaBSE embeddings to find sentence pairs that are similar in meaning 
and an alignment algorithm based on Dijkstra's algorithm to find the optimal alignment.

### License 
Copyright 2023 Steinþór Steingrímsson
SentAlign is released under the [Apache License, Version 2.0](LICENSE).


### Building the environment
```
The environment can be built using the provided environment.yml file:
conda env create -f environment.yml
```

### Running the alignment
We assume that the files to align have the same names in the source and target language, but are kept in folders named after the language. For example, if we want to align the files in the folder `/path/to/files/to/align` we would have the following structure:
```
/path/to/files/to/align/eng/file1.txt
/path/to/files/to/align/eng/file2.txt
...
/path/to/files/to/align/isl/file1.txt
/path/to/files/to/align/isl/file2.txt
```

To run the alignment you first need to activate the environment:
```
conda activate SentAlign
```

Start by creating a list of files to align:


python3 files2align.py -dir /path/to/files/to/align --source-language eng

Then you run the alignments. Aligning English and Icelandic files:

python3 sentAlign.py -dir /path/to/files/to/align -sl eng -tl isl
