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
```
conda activate SentAlign
```
Start by creating a list of files to align:
python3 files2align.py -dir /path/to/files/to/align

Then you run the alignments. Aligning English and Icelandic files:

python3 sentAlign.py -dir /path/to/files/to/align -sl eng -tl isl
