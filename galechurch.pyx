# cython: language_level=3

"""
A Cython implementation of the Gale-Church algorithm
"""
import sys

cimport numpy as np
cimport cython
from libc.math cimport log, exp, sqrt, isinf

from os import remove
from os.path import basename
from file_read_back import FileReadBackwards
import tempfile

cdef long score_cutoff
cdef int load_to_memory_threshold
cdef str alignment_type
cdef int num_proc
cdef int mean_xy
cdef float variance_xy
cdef double LOG2

score_cutoff = 10000000
load_to_memory_threshold = 1000
alignment_type = 'pharaoh'
num_proc = 1

# Alignment costs: -100*log(p(x:y)/p(1:1))
bead_costs = {
     (2, 2): 440,
     (2, 1): 230,
     (1, 2): 230,
     (1, 1): 0,
     (1, 0): 450,
     (0, 1): 450
}

# Length cost parameters
mean_xy = 1
variance_xy = 6.8
LOG2 = log(2)



cdef norm_cdf(double z):
    """ Cumulative distribution for N(0, 1) """
    cdef double t
    t = 1 / (1 + 0.2316419 * z)
    return (1 - 0.3989423 * exp(-z * z / 2) *
            ((((1.330274429 * t - 1.821255978) * t
               + 1.781477937) * t - 0.356563782) * t + 0.319381530) * t)


cdef norm_logsf(double z):
    """ Logarithm of the survival function for N(0, 1) """
    try:
        return log(1 - norm_cdf(z))
    except ValueError:
        return float('-inf')


cdef length_cost(sx, sy):
    """ -100*log[p(|N(0, 1)|>delta)] """
    cdef long lx, ly
    cdef double delta
    lx, ly = sum(sx), sum(sy)
    m = (lx + ly * mean_xy) / 2
    try:
        delta = (lx - ly * mean_xy) / sqrt(m * variance_xy)
    except ZeroDivisionError:
        return float('-inf')
    return - 100 * (LOG2 + norm_logsf(abs(delta)))


cdef calc_cost(int i, int j, x, y, m):
    costs = []
    if i == j == 0:
        return (0,0,0)
    else:
        costs.append(min((m[i - di, j - dj][0] + length_cost(x[i - di:i], y[j - dj:j]) + bead_cost, di, dj) for (di, dj), bead_cost in bead_costs.items() if i - di >= 0 and j - dj >= 0))
    return min(costs)


cdef calc_cost_large(int i, int j, x, y, scorecutoff, m):
    costs = []
    if i == j == 0:
        return (0,0,0)
    else:
        try:
            costs.append(min((m[i - di][j - dj][0] + length_cost(x[i - di:i], y[j - dj:j]) + bead_cost, di, dj) for (di, dj), bead_cost in bead_costs.items() if i - di >= 0 and j - dj >= 0))
        except Exception as e:
            print(e)
            print(scorecutoff, i, j)
            return (scorecutoff, i, j)
    return min(costs)


def _align(x, y, int longlength=1000, scorecutoff=None, my_basename='filename'):
    m = {}
    cdef int max_length
    cdef long highest_score
    cdef int i, j, currline
    cdef long c, di, dj

    max_length = max(len(x), len(y))
    temp = tempfile.NamedTemporaryFile(prefix='gc_' + my_basename + '_', delete=False, mode='w')
    highest_score = 0

    if (max_length > longlength):
            for i in range(len(x)+1):
                m[i] = {}
                for j in range(len(y)+1):
                    min_i_j = calc_cost_large(i, j, x, y, scorecutoff, m)
                    m[i][j] = min_i_j

                for key, value in m[i].items():
                    if not isinf(value[0]):
                        if scorecutoff is None:
                            temp.write(str(key) + '|' + str(int(value[0])) + '|' + str(value[1]) + '|' + str(value[2]) + '\t')
                        elif int(value[0]) < scorecutoff:
                            temp.write(str(key) + '|' + str(int(value[0])) + '|' + str(value[1]) + '|' + str(value[2]) + '\t')
                temp.write('\n')
                if i - 3 >= 0:
                    m[i - 3] = {}
    temp.close()

    if (max_length > longlength):
        i, j = len(x), len(y)
        try:
            with FileReadBackwards(temp.name) as fi:
                tempdict = {}
                currLine = i
                for line in fi:
                    di = 0
                    if currLine == i:
                        currvalues = line.strip().split('\t')
                        for cv in currvalues:
                            v = cv.split('|')
                            try:
                                tempdict[int(v[0])] = (int(v[1]), int(v[2]), int(v[3]))
                            except: #length cost was too large for all combinations
                                tempdict[0] = (100000000, 1, 0)
                        while di == 0:
                            try:
                                (c, di, dj) = tempdict[j]
                            except: #Exception for the case of length cost having been cut off when generating the temp file
                                (c, di, dj) = (100000000, 0, 1)
                            if c > highest_score:
                                highest_score = c
                            if di == dj == 0:
                                break
                            yield (i - di, i), (j - dj, j), highest_score, len(x), len(y)
                            i -= di
                            j -= dj
                        if di <= dj <= 0:
                            break

                    currLine -= 1
        except Exception as e:
            print(e)
            print('error in _align')
            sys.exit(1)
            print('finished fi')
    else:
        for i in range(len(x)+1):
            for j in range(len(y)+1):
                m[i, j] = calc_cost(i, j, x, y, m)

        while True:
            (c, di, dj) = m[i, j]
            if c > highest_score:
                highest_score = c
            if di == dj == 0:
                break
            yield (i-di, i), (j-dj, j), highest_score, len(x), len(y)
            i -= di
            j -= dj
    remove(temp.name)


cdef char_length(sentence):
    """ Length of a sentence in characters """
    return len(sentence.replace(' ', ''))


def align(sx, sy, int longlength, scorecutoff, my_basename):
    """ Align two groups of sentences """
    cx = list(map(char_length, sx))
    cy = list(map(char_length, sy))
    cdef long highest_score
    cdef int lenx, leny
    for (i1, i2), (j1, j2), highest_score, len_x, len_y in reversed(list(_align(cx, cy, longlength, scorecutoff, my_basename))):
        source_sentences = range(i1, i2)
        target_sentences = range(j1, j2)
        if alignment_type == 'pharaoh':
            yield str(list(source_sentences)) + ':' + str(list(target_sentences)), highest_score, len_x, len_y
        elif alignment_type == 'text':
            yield ' '.join(sx[i1:i2]) + '\t' + ' '.join(sy[j1:j2]), highest_score, len_x, len_y


def read_blocks(f):
    # Blocks are separated by an empty line. They can be paragraphs or documents.
    block = []
    for l in f:
        if not l.strip():
            yield block
            block = []
        else:
            block.append(l.strip())
    if block:
        yield block


def gale_church(corpus_x, corpus_y, int longlength, scorecutoff):
    alignments_out = ''
    with open(corpus_x) as fx, open(corpus_y) as fy:
        for block_x, block_y in zip(read_blocks(fx), read_blocks(fy)):
            for alignment, highest_score, len_x, len_y in align(block_x, block_y, longlength, scorecutoff, basename(corpus_x)):
                alignments_out += alignment + '\n'
    #print('output', alignments_out, highest_score, len_x, len_y)
    return alignments_out, highest_score, len_x, len_y
