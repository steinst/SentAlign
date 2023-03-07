# cython: language_level=3

"""
A Cython implementation of the Gale-Church algorithm
"""
cimport cython
import numpy as np
cimport numpy as np


def calculate_anchor_nomatrix_set(galechurch_alignments, src_emb_dict, trg_emb_dict, float minimum_anchor_score,
                                  anchor_source_list_lines, anchor_target_list_lines, int source_len, int target_len,
                                  anchor_source_list, anchor_target_list, int start_source, int end_source,
                                  int start_target, int end_target, labse_score_matrix,
                                  source_loc_start_list, source_loc_end_list, target_loc_start_list, target_loc_end_list):
    cdef int i_ctr
    cdef int j_ctr
    cdef str i, j
    cdef int source_loc_start, source_loc_end, target_loc_start, target_loc_end
    cdef float labse_score

    labse_alignments = []
    i_ctr = 0
    for i in anchor_source_list:
        j_ctr = 0
        source_loc_start = source_loc_start_list[i_ctr]
        source_loc_end = source_loc_end_list[i_ctr]
        if source_loc_start >= start_source and source_loc_end < end_source:
            for j in anchor_target_list:
                target_loc_start = target_loc_start_list[j_ctr]
                target_loc_end = target_loc_end_list[j_ctr]
                if target_loc_start >= start_target and target_loc_end < end_target:
                    labse_score = labse_score_matrix[i_ctr][j_ctr]
                    if labse_score > minimum_anchor_score:
                        labse_alignments.append([anchor_source_list_lines[i_ctr].strip(), anchor_target_list_lines[j_ctr].strip()])
                j_ctr += 1
        i_ctr += 1

    a_list = [value for value in galechurch_alignments if value in labse_alignments] + [[str(source_len), str(target_len)]]
    return a_list


def calculate_anchor_set(source_start_matrix, source_end_matrix, target_start_matrix, target_end_matrix,
                         galechurch_alignments, labse_matrix, double minimum_ancor_score, anchor_source_list_lines,
                         anchor_target_list_lines, int start_source, int end_source, int start_target, int end_target):

    cdef int i_ctr, j_ctr
    labse_alignments = []

    above_threshold = np.where(labse_matrix > minimum_ancor_score)

    for i_ctr, j_ctr in zip(above_threshold[0], above_threshold[1]):
        if source_start_matrix[i_ctr] > start_source + 2 and source_end_matrix[i_ctr] < end_source - 2:
            if target_start_matrix[j_ctr] > start_target + 2 and target_end_matrix[j_ctr] < end_target - 2:
                labse_alignments.append([anchor_source_list_lines[i_ctr].strip(), anchor_target_list_lines[j_ctr].strip()])

    anchor_list = [value for value in galechurch_alignments if value in labse_alignments] + [[str(end_source), str(end_target)]]
    return anchor_list
