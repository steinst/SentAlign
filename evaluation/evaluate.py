#!/usr/bin/env python3

"""
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
        and on the target side
    - a gold alignment is considered correct if it overlaps with a test alignment on the source side

"""

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
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
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
        fn = len(list(goldalign - set(goldaligns_in_test)))

    return np.array([tpstrict, fpstrict, tplax, fplax, fn], dtype=np.int32)


def score_multiple(gold_list, test_list, keep_nulls, value_for_div_by_0=0.0):
    # accumulate counts for all gold/test files
    counts = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    tp_fp = 0
    tp_fn = 0
    # rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    #print(len(test_list[0]))
    for goldalign, testalign in zip(gold_list, test_list):
        if keep_nulls:
            testalign = [(x, y) for x, y in testalign if len(x) and len(y)]
            goldalign = [(x, y) for x, y in goldalign if len(x) and len(y)]

        counts += _precision(goldalign=goldalign, testalign=testalign)
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

    if counts[0] + counts[1] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = counts[0] / float(tp_fn)

    if counts[2] + counts[3] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = counts[2] / float(counts[2] + counts[4])

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

    args = parser.parse_args()
    #print(args.nulls)

    if len(args.test) != len(args.gold):
        raise Exception('number of gold/test files must be the same')

    gold_list = [read_alignments(x) for x in args.gold]
    test_list = [read_alignments(x) for x in args.test]

    res = score_multiple(gold_list=gold_list, test_list=test_list, keep_nulls=args.nulls)
    log_final_scores(res)


if __name__ == '__main__':
    main()
