"""The large-file anchoring of SentAlign 1.0 ("greedy" anchors), kept bug for bug for ``--anchors legacy-greedy``
(paper-2023 mode uses it for files above ``-maxgc`` sentences, as version 1 did).

KNOWN FAULTS, kept on purpose so that old corpora can be regenerated:
* ``greedy_anchor_selection_large`` (greedy.pyx) tests the score threshold on one cell and reads the score
  from another, 20 % earlier in the matrix; the anchor is in effect the best cell near the diagonal of a shifted
  region, not a confirmed match;
* the length filter of the candidates has no effect (``bool('False')``);
* the score matrices are filled cell by cell in Python (slow).
Version 2 uses ``sentalign.anchors`` instead.

The mid-size anchoring of version 1 (Gale-Church beads confirmed by a LaBSE matrix) is not here: in the released
code it could never work (TypeError, out-of-bounds read, integer fallback matrix) and ended either in a
segmentation fault or in solving the whole graph.  paper-2023 mode now solves the whole graph directly.
"""
import logging

import numpy as np

from sentalign.legacy.greedy import get_highest_labse_anchor, greedy_anchor_selection_large
from sentalign.scoring.embedding import anchor_lists, text_dict

log = logging.getLogger('sentalign.legacy')

MAX_MATRIX_SIZE_FOR_GREEDY = 1000000000


def _pos(x):
    """First number of an anchor coordinate: ``int(x.split(',')[0])`` for strings, the value itself otherwise."""
    try:
        return int(x.split(',')[0])
    except (AttributeError, ValueError):
        return x


def legacy_greedy_anchors(src_side, trg_side, scorer, node_limit, minimum_labse_anchor=0.95, labse_subtraction=0.05,
                          cutoff_penalty=1.25):
    """-> [(i, j), ...] including the end of the document as the last anchor, as version 1 produced them."""
    source_len, target_len = len(src_side), len(trg_side)
    anchor_list = [[str(source_len), str(target_len)]]
    # version 1 create_anchor_files(file, 1, True): the single sentences, texts and line numbers with newline
    anchor_source_list, anchor_source_list_lines = anchor_lists(src_side, 1)
    anchor_target_list, anchor_target_list_lines = anchor_lists(trg_side, 1)
    anchor_src_emb_dict = text_dict(src_side, scorer.src_emb, 1)
    anchor_trg_emb_dict = text_dict(trg_side, scorer.trg_emb, 1)

    if (source_len * target_len) > MAX_MATRIX_SIZE_FOR_GREEDY:
        slow_greedy = True
        while slow_greedy:
            new_anchor_list = []
            slow_greedy = False
            start_source = 0
            start_target = 0
            for anchor in anchor_list:
                if (int(anchor[0]) - int(start_source)) * (int(anchor[1]) - int(start_target)) > MAX_MATRIX_SIZE_FOR_GREEDY:
                    new_anchor = get_highest_labse_anchor(start_source, start_target, anchor, anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict)
                    if new_anchor is not None:
                        new_anchor_list.append(new_anchor)
                        try:
                            new_matrix_first = (int(new_anchor[0]) - start_source) * (int(new_anchor[1]) - start_target)
                            new_matrix_second = (int(anchor[0]) - int(new_anchor[0])) * (int(anchor[1]) - int(new_anchor[1]))
                            if (new_matrix_first > MAX_MATRIX_SIZE_FOR_GREEDY) or (new_matrix_second > MAX_MATRIX_SIZE_FOR_GREEDY):
                                slow_greedy = True
                        except (TypeError, ValueError, IndexError):
                            log.warning('slow greedy: could not compute the size of the new segments', exc_info=True)
                    else:
                        slow_greedy = True
                new_anchor_list.append(anchor)
                start_source = int(anchor[0])
                start_target = int(anchor[1])
            anchor_list = new_anchor_list.copy()
            log.debug('slow greedy anchor list: %s', anchor_list)

    # version 1 greedy_procedure_large
    file_minimum_anchor_score = minimum_labse_anchor
    temp_cutoff4anchoring = node_limit
    maxNumberOfKnots = 9223372036854775807
    while maxNumberOfKnots > temp_cutoff4anchoring:
        start_source = 0
        start_target = 0
        new_anchor_list = []
        maxNumberOfKnots = 0
        for anchor in anchor_list:
            end_source = _pos(anchor[0])
            end_target = _pos(anchor[1])
            current_source_length = end_source - start_source
            current_target_length = end_target - start_target

            labse_score_matrix = np.zeros((current_source_length, current_target_length), dtype=np.float64)
            i_ctr = 0
            for i in anchor_source_list:
                if i_ctr >= start_source and i_ctr < end_source:
                    j_ctr = 0
                    for j in anchor_target_list:
                        if j_ctr >= start_target and j_ctr < end_target:
                            labse_score_matrix[i_ctr - start_source][j_ctr - start_target] = np.dot(anchor_src_emb_dict[i.strip()], anchor_trg_emb_dict[j.strip()])
                        j_ctr += 1
                i_ctr += 1

            numberOfKnots = (end_source - start_source) * (end_target - start_target)
            if numberOfKnots > maxNumberOfKnots:
                maxNumberOfKnots = numberOfKnots
            if numberOfKnots > temp_cutoff4anchoring:  # (30k x 30k matrix or equivalent)
                temp_greedy_anchor = greedy_anchor_selection_large(file_minimum_anchor_score, labse_score_matrix)
                if temp_greedy_anchor is not None:
                    greedy_anchor = [str(temp_greedy_anchor[0] + start_source), str(temp_greedy_anchor[1] + start_target)]
                    if greedy_anchor not in new_anchor_list:
                        new_anchor_list.append(greedy_anchor)
            if anchor not in new_anchor_list:
                new_anchor_list.append(anchor)
            start_source = int(_pos(anchor[0]))
            start_target = int(_pos(anchor[1]))
        anchor_list = new_anchor_list.copy()
        temp_cutoff4anchoring = temp_cutoff4anchoring * cutoff_penalty
        file_minimum_anchor_score = file_minimum_anchor_score - labse_subtraction
    return [(int(_pos(a[0])), int(_pos(a[1]))) for a in anchor_list]
