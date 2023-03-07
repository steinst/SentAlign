SentAlign is a sentence alignment tool for parallel corpora.

It uses LaBSE embeddings to find sentence pairs that are similar in meaning and an alignment algorithm based on Dijkstra's algorithm to find the optimal alignment.

Start by creating a list of files to align:
python3 files2align.py -dir /path/to/files/to/align

Then you run the alignments. Aligning English and Icelandic files, using 12 processes would look like this:

python3 sentAlign.py -dir /path/to/files/to/align -proc 12 -sl eng -tl isl
