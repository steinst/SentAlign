# cython: language_level=3

cimport cython
import numpy as np
from cython.view cimport array as cvarray
#cimport numpy as np

cdef extern from "Python.h":
    object PyUnicode_AsUTF8String(object unicode)

@cython.boundscheck(False)
@cython.wraparound(False)
def create_labse_score_matrix(anchor_source_list: list[str], anchor_target_list: list[str], src_emb_dict: dict, trg_emb_dict: dict):
    cdef str i, j
    cdef int i_ctr, j_ctr
    cdef double[:,:] labse_score_matrix = np.zeros((len(anchor_source_list), len(anchor_target_list)), dtype=np.float64)
    #out_matrix = np.array([[0 for x in range(len(anchor_target_list))] for y in range(len(anchor_source_list))])

    i_ctr = 0
    for i in anchor_source_list:
        j_ctr = 0
        for j in range(anchor_target_list):
        #atl_length = len(anchor_target_list)
        #for j_ctr in prange(0, atl_length, 1, nogil=True):
            try:
                labse_score_matrix[i_ctr,j_ctr] = trg_emb_dict[j.strip()].dot(src_emb_dict[i.strip()].transpose())
            except Exception as e:
                print(e)
                labse_score_matrix[i_ctr,j_ctr] = 0
            j_ctr = j_ctr + 1
        i_ctr = i_ctr + 1
    return labse_score_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
def loc_start_end_matrices(anchor_lines: list[str]):
    cdef int i
    cdef list[int] start_list, end_list
    start_list = []
    end_list = []
    for i in range(0, len(anchor_lines)):
        start_list.append(int(anchor_lines[i].split(',')[0]))
        end_list.append(int(anchor_lines[i].split(',')[1]))
    return start_list, end_list