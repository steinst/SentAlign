# -*- coding: UTF-8 -*-

import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-folder', '-dir')
parser.add_argument('--source-language', '-sl', default='eng')
parser.add_argument('--output-folder', '-out', default='output')
args = parser.parse_args()

corpus_folder = args.corpus_folder
output_folder = args.corpus_folder + '/' + args.output_folder
source_language_folder = corpus_folder + '/' + args.source_language

files_to_process = [f for f in listdir(source_language_folder) if isfile(join(source_language_folder, f))]
#files_not_to_process = [f.replace('.aligned', '').replace('overlaps.', '') for f in listdir(output_folder) if
#                        f.endswith('.aligned')]

with open(corpus_folder + '/files2align.txt', 'w') as f:
    for file in files_to_process:
        f.write(file + '\n')

with open(corpus_folder + '/filesdone.txt', 'w') as f:
    for file in files_not_to_process:
        f.write(file + '\n')
