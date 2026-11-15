#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

This file is based on score.py of Vecalign (https://github.com/thompsonb/vecalign); it was adapted for SentAlign
(the scoring of null alignments, the command line).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import sys
from collections import defaultdict
from ast import literal_eval

import numpy as np

"""
Reimplementation of lax and strict precision and recall, based on
   https://www.aclweb.org/anthology/W11-4624/ and [Vecalign]

Precision is the number of correct alignments divided by the number of alignments in the test set.
Recall is calculated as the number of correct alignments divided by the number of gold alignments.
Lax precision and recall are calculated as precision and recall, but with the following relaxation:

    - a test alignment is considered correct if it overlaps with a gold alignment on the source side
        and on the target side (lax precision)
    - a gold alignment is considered found if it overlaps with a test alignment on the source side
        and on the target side (lax recall; as the evaluation of Bleualign and score.py of Vecalign)

The evaluation script of SentAlign 1.0 (the 2023 paper) computed lax recall differently: lax true positives of the
test alignments / (those + gold alignments without an exact match).  A gold alignment that was only found under
the lax condition counted as missed there, so lax recall and lax F1 came out lower (German-French 0.933 / 0.964
instead of 0.980 / 0.989).  ``--lax-recall paper-2023`` (``lax_recall='paper-2023'``) gives those numbers; strict
scores and lax precision are the same in both.
"""

LAX_RECALL = ('standard', 'paper-2023')


def read_alignments(fin):
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))

    # I know bluealign files have a few entries entries missing,
    # but I don't fix them in order to be consistent previous reported scores
    return alignments


def _precision(goldalign, testalign):
    """
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments, and the number of gold alignments
    without an exact match (the "false negatives" of the 2023 lax recall)
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

    goldaligns_in_test = []
    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            # strict match
            tpstrict += 1
            tplax += 1
            goldaligns_in_test.append((test_src, test_target))
        else:
            # For anything with partial gold/test overlap on the source,
            #   see if there is also partial overlap on the gold/test target
            # If so, its a lax match
            target_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    target_ids.add(tgt_id)
            if set(test_target).intersection(target_ids):
                fpstrict += 1
                tplax += 1
                goldaligns_in_test.append((test_src, test_target))
            else:
                fpstrict += 1
                fplax += 1
    # paper-2023 lax recall only: the gold alignments that are not among the matched *test* alignments, i.e. those
    # without an exact match
    fn = len(goldalign - set(goldaligns_in_test))

    return np.array([tpstrict, fpstrict, tplax, fplax, fn], dtype=np.int32)


def score_multiple(gold_list, test_list, keep_nulls, value_for_div_by_0=0.0, lax_recall='standard'):
    """Scores over all gold/test files together.  `keep_nulls` True (the default of the command line) leaves the
    null alignments out.  `lax_recall`: 'standard' (Bleualign, Vecalign) or 'paper-2023' (see the top of the file)."""
    if lax_recall not in LAX_RECALL:
        raise ValueError('lax_recall must be one of %s (got %r)' % (', '.join(LAX_RECALL), lax_recall))
    # accumulate counts for all gold/test files
    counts = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    tp_fp = 0
    tp_fn = 0
    # rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    #print(len(test_list[0]))
    for goldalign, testalign in zip(gold_list, test_list):
        if keep_nulls:
            testalign = [(x, y) for x, y in testalign if len(x) and len(y)]
            goldalign = [(x, y) for x, y in goldalign if len(x) and len(y)]

        counts += _precision(goldalign=goldalign, testalign=testalign)
        # recall is precision with the arguments swapped (on the same alignments as the other scores)
        rcounts += _precision(goldalign=testalign, testalign=goldalign)[:4]
        tp_fp += len(testalign)
        tp_fn += len(goldalign)

        # recall is precision with no insertion/deletion and swap args
        #test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        #gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        #print(testalign)
        #print(test_no_del)
        #rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    # print(counts)
    #print(tp_fp)
    #print(tp_fn)
    if counts[0] + counts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = counts[0] / float(tp_fp)

    if counts[2] + counts[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = counts[2] / float(counts[2] + counts[3])

    if counts[0] + counts[1] == 0 or tp_fn == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = counts[0] / float(tp_fn)

    if lax_recall == 'paper-2023':
        if counts[2] + counts[3] == 0 or counts[2] + counts[4] == 0:
            rlax = value_for_div_by_0
        else:
            rlax = counts[2] / float(counts[2] + counts[4])
    elif rcounts[2] + rcounts[3] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

    if (pstrict + rstrict) == 0:
        fstrict = value_for_div_by_0
    else:
        fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

    if (plax + rlax) == 0:
        flax = value_for_div_by_0
    else:
        flax = 2 * (plax * rlax) / (plax + rlax)

    result = dict(recall_strict=rstrict,
                  recall_lax=rlax,
                  precision_strict=pstrict,
                  precision_lax=plax,
                  f1_strict=fstrict,
                  f1_lax=flax)

    return result


def log_final_scores(res):
    print(' ---------------------------------', file=sys.stderr)
    print('|             |  Strict |    Lax  |', file=sys.stderr)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=sys.stderr)
    print(' ---------------------------------', file=sys.stderr)


EVALUATION_SETS = {
    'bleualign': ('the German-French Text+Berg evaluation set (Bleualign)',
                  'Sennrich, R. and Volk, M. (2010). MT-based Sentence Alignment for OCR-generated Parallel Texts. AMTA 2010.',
                  '''@inproceedings{Sennrich2010MTbasedSA,
    title = {{MT-based Sentence Alignment for OCR-generated Parallel Texts}},
    author = "Rico Sennrich and Martin Volk",
    booktitle = "Proceedings of the 9th Conference of the Association for Machine Translation in the Americas: Research Papers",
    address = "Denver, Colorado",
    year = "2010",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2010.amta-papers.14",
}'''),
    'parice': ('the English-Icelandic ParIce evaluation set',
               'Barkarson, S. and Steingrímsson, S. (2019). Compiling and Filtering ParIce: An English-Icelandic '
               'Parallel Corpus. NoDaLiDa 2019; and the SentAlign paper, where the evaluation set was first used.',
               '''@inproceedings{barkarson-steingrimsson-2019-compiling,
    title = {{Compiling and Filtering {P}ar{I}ce: An {E}nglish-{I}celandic Parallel Corpus}},
    author = "Barkarson, Starka{\\dh}ur and Steingr{\\'\\i}msson, Stein{\\th}{\\'o}r",
    booktitle = "Proceedings of the 22nd Nordic Conference on Computational Linguistics",
    year = "2019",
    address = "Turku, Finland",
    publisher = {Link{\\"o}ping University Electronic Press},
    url = "https://aclanthology.org/W19-6115",
    pages = "140--145",
}'''),
}


def print_citations(gold_files, bibtex=False):
    """Say which publication to cite for the evaluation set that was used (recognised by the folder name; both
    are named when it cannot be recognised)."""
    names = [k for k in EVALUATION_SETS if any(k in f.lower() for f in gold_files)] or list(EVALUATION_SETS)
    print('\nIf you report these results, please cite the evaluation set:', file=sys.stderr)
    for k in names:
        print('  [%s] %s' % EVALUATION_SETS[k][:2], file=sys.stderr)
    if bibtex:
        print('\nBibTeX:\n', file=sys.stderr)
        for k in names:
            print(EVALUATION_SETS[k][2] + '\n', file=sys.stderr)
    else:
        print('(--bibtex prints the BibTeX entries)', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        'Compute strict/lax precision and recall for one or more pairs of gold/test alignments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--test', type=str, nargs='+', required=True,
                        help='one or more test alignment files')

    parser.add_argument('-g', '--gold', type=str, nargs='+', required=True,
                        help='one or more gold alignment files')

    parser.add_argument('-n', '--nulls', action='store_false',
                        help='use null alignments')

    parser.add_argument('--lax-recall', choices=LAX_RECALL, default='standard',
                        help='standard: a gold alignment is found if a test alignment overlaps it on both sides '
                             '(Bleualign, Vecalign); paper-2023: the computation of the SentAlign 1.0 script, for '
                             'the numbers of the 2023 paper')

    parser.add_argument('--bibtex', action='store_true',
                        help='also print the BibTeX entries of the evaluation sets to cite')

    args = parser.parse_args()
    #print(args.nulls)

    if len(args.test) != len(args.gold):
        raise Exception('number of gold/test files must be the same')

    gold_list = [read_alignments(x) for x in args.gold]
    test_list = [read_alignments(x) for x in args.test]

    res = score_multiple(gold_list=gold_list, test_list=test_list, keep_nulls=args.nulls, lax_recall=args.lax_recall)
    log_final_scores(res)
    if args.lax_recall == 'paper-2023':
        print('lax recall and F1 as computed for the 2023 paper (not comparable with Bleualign / Vecalign scores)',
              file=sys.stderr)
    print_citations(args.gold, args.bibtex)


if __name__ == '__main__':
    main()
