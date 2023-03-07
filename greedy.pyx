# cython: language_level=3

"""
A Cython implementation of the Gale-Church algorithm
"""

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel cimport prange

## add greedy anchor selection for very large files (e.g. over 40 k sentences) then not use labse_score_matrix but calculate labse on the fly


@cython.boundscheck(False)
@cython.wraparound(False)
def get_highest_labse_anchor(start_source: int, start_target: int, anchor: list[str], anchor_source_list: list[str], anchor_target_list: list[str], anchor_source_list_lines: list[str], anchor_target_list_lines: list[str], src_emb_dict, trg_emb_dict):
    cdef int end_source, end_target, source_len, target_len, source_ctr, target_ctr
    cdef int min_target, max_target
    cdef float maximum_source_target_difference
    cdef double labse_score, highest_score
    cdef highest_anchor

    highest_score = 0.0
    highest_anchor = None
    try:
        end_source = int(anchor[0].split(',')[0])
    except:
        end_source = anchor[0]
    try:
        end_target = int(anchor[1].split(',')[0])
    except:
        end_target = anchor[1]

    source_len = end_source - start_source
    target_len = end_target - start_target
    # not start in the beginning to try to make larger spans
    start_source = int(start_source + (0.2 * source_len))
    start_target = int(start_target + (0.2 * target_len))

    maximum_source_target_difference = max(int(1.15*(abs(source_len-target_len))), int(0.15*((target_len+source_len)/2)))

    for source_ctr in range(start_source, end_source-2):
        #þetta ræður bara við fylki þar sem allar linur eru með og þær eru í röð
        if start_source + 2 < int(anchor_source_list_lines[source_ctr].strip().split(',')[0]) < end_source - 2:
            min_target = max(start_target, min(int(source_ctr-maximum_source_target_difference), end_target))
            max_target = min(end_target, max(start_target, int(source_ctr+maximum_source_target_difference)))
            for target_ctr in range(min_target+2, max_target-2):
                if min_target + 2 < int(anchor_target_list_lines[target_ctr].strip().split(',')[0]) < max_target - 2:
                    try:
                        labse_score = trg_emb_dict[anchor_target_list[target_ctr].strip()].dot(src_emb_dict[anchor_source_list[source_ctr].strip()].transpose())
                    except Exception as e:
                        labse_score = 0
                    try:
                        if labse_score > highest_score:
                            highest_score = labse_score
                            highest_anchor = [str(anchor_source_list_lines[source_ctr]).strip(), str(anchor_target_list_lines[target_ctr]).strip()]
                    except Exception as e:
                        print(e)
    return highest_anchor


def greedy_anchor_selection(int start_source, int start_target, anchor,
                            anchor_source_list_lines, anchor_target_list_lines, float minimum_score, np.ndarray[double, ndim=2] labse_score_matrix):
    cdef int end_source, end_target, source_len, target_len, i_ctr, j_ctr
    cdef int min_target, max_target
    cdef float maximum_source_target_difference
    cdef double labse_score, highest_score
    cdef int source_loc_start, source_loc_end, target_loc_start, target_loc_end

    cdef highest_anchor

    highest_score = 0.0
    highest_anchor = None
    try:
        end_source = int(anchor[0].split(',')[0])
    except:
        end_source = anchor[0]
    try:
        end_target = int(anchor[1].split(',')[0])
    except:
        end_target = anchor[1]

    source_len = end_source - start_source
    target_len = end_target - start_target
    # not start in the beginning to try to make larger spans
    start_source = int(start_source + (0.2 * source_len))
    start_target = int(start_target + (0.2 * target_len))

    maximum_source_target_difference = max(int(1.15*(abs(source_len-target_len))), int(0.15*((target_len+source_len)/2)))
    above_threshold = np.where(labse_score_matrix[start_source:end_source, start_target:end_target] > minimum_score)
    for i_ctr, j_ctr in zip(above_threshold[0], above_threshold[1]):
        if int(anchor_source_list_lines[i_ctr]) > start_source + 2 and int(anchor_source_list_lines[i_ctr]) < end_source - 2:
            min_target = max(start_target, min(int(i_ctr-maximum_source_target_difference), end_target))
            max_target = min(end_target, max(start_target, int(i_ctr+maximum_source_target_difference)))
            if int(anchor_target_list_lines[j_ctr]) > min_target + 2 and int(anchor_target_list_lines[j_ctr]) < max_target - 2:
                labse_score = labse_score_matrix[i_ctr, j_ctr]
                if labse_score > highest_score:
                    highest_score = labse_score
                    highest_anchor = [str(anchor_source_list_lines[i_ctr]).strip(), str(anchor_target_list_lines[j_ctr]).strip()]
    return highest_anchor

@cython.boundscheck(False)
@cython.wraparound(False)
def greedy_anchor_selection_large(double minimum_score, np.ndarray[double, ndim=2] labse_score_matrix):
    cdef int source_len, target_len, i_ctr, j_ctr
    cdef int min_target, max_target
    cdef float maximum_source_target_difference
    cdef double labse_score, highest_score
    cdef highest_anchor
    #cdef int source_loc_start, source_loc_end, target_loc_start, target_loc_end

    highest_score = 0
    highest_anchor = None


    source_len = len(labse_score_matrix)
    target_len = len(labse_score_matrix[0])

    start_source = int(0.2 * source_len)
    start_target = int(0.2 * target_len)


    maximum_source_target_difference = max(int(1.15*(abs(source_len-target_len))), int(0.15*((target_len+source_len)/2)))


    above_threshold = np.where(labse_score_matrix[start_source:source_len-1, start_target:target_len-1] > minimum_score)
    for i_ctr, j_ctr in zip(above_threshold[0], above_threshold[1]):
        if i_ctr > start_source + 2 and i_ctr < source_len - 2:
            min_target = max(start_target, min(int(i_ctr-maximum_source_target_difference), target_len))
            max_target = min(target_len, max(start_target, int(i_ctr+maximum_source_target_difference)))
            if j_ctr > min_target + 2 and j_ctr < max_target - 2:
                labse_score = labse_score_matrix[i_ctr, j_ctr]
                if labse_score > highest_score:
                    highest_score = labse_score
                    highest_anchor = [i_ctr,j_ctr]
    return highest_anchor