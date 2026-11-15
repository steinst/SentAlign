# -*- coding: UTF-8 -*-

from tokenizer import split_into_sentences
import sys
import os
import multiprocessing
from multiprocessing.pool import Pool

foldername = sys.argv[1]
outfolder = sys.argv[2]


def tokenize_is(curr_file):
    inFile = open(curr_file, 'r')
    inLines = inFile.readlines()
    inFile.close()

    outfile = curr_file.replace(foldername, outfolder) + '.sentsplit'

    with open(outfile, 'w') as is_out:
        for currLine in inLines:
            line_out = ''
            g = split_into_sentences(currLine)
            if len(currLine.strip()) < 1:
                is_out.write(line_out + '\n')
            else:
                for sentence in g:
                    tokens = sentence.split()
                    if len(tokens) >= 12:
                        lasttokens = tokens[-12:]
                    else:
                        lasttokens = tokens
                    currplace = 0
                    for token in lasttokens:
                        currplace = currLine.find(token,currplace) + len(token)
                    current_out = currLine[:currplace]
                    currLine = currLine[currplace:]
                    line_out += current_out.lstrip().strip() + '\n'

                is_out.write(line_out)

if __name__ == "__main__":
    filenames = []
    for subdir, dirs, files in os.walk(foldername):
        for name in files:
            if name.endswith(".txt"):
                os.makedirs(subdir.replace(foldername, outfolder), exist_ok=True)
                filenames.append(subdir + '/' + name)
                #tokenize_is(subdir + '/' + name)
    with Pool(20) as p:
        p.map(tokenize_is, filenames)
