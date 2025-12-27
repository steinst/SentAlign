# -*- coding: UTF-8 -*-
import multiprocessing

from reportInfo import ReportInfo
from repeatedTimer import RepeatedTimer
import numpy as np
import argparse
import torch
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
import glob
import re
import os
from itertools import cycle
import datetime
import string
import random
import time

from galechurch import gale_church
from greedy import greedy_anchor_selection, get_highest_labse_anchor, greedy_anchor_selection_large
from anchoring import calculate_anchor_nomatrix_set, calculate_anchor_set
from utilities import create_labse_score_matrix, loc_start_end_matrices
from align_anchors import align_anchors_multi


os.environ["TOKENIZERS_PARALLELISM"] = "false"

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')

parser = argparse.ArgumentParser()
#Input and output file locations
parser.add_argument('--corpus-folder', '-dir')
parser.add_argument('--source-language', '-sl', default='eng')
parser.add_argument('--target-language', '-tl', default='isl')
parser.add_argument('--filename', '-f', help='Name of source and target file(s) to be aligned', type=str, nargs='+')
parser.add_argument('--output-folder', '-out', default='output')
#Aligner settings
parser.add_argument('--max-concatenations', '-concats', type=int, help='Maximum number of concatenated sentences per language', default=4)
parser.add_argument('--free-concatenations', '-freejoins', type=int, help='Maximum number of concatenations before penalty is applied', default=2)
parser.add_argument('--score-cutoff', '-cutoff', type=float, help='Minimum similarity score for a sentence pair to be considered', default=0.4)
parser.add_argument('--minimum-length-words', '-minwords', type=int, help='Minimum number of words per language, for a sentence pair to be considered', default=1)
parser.add_argument('--maximum-length-words', '-maxwords', type=int, help='Maximum number of words per language, for a sentence pair to be considered', default=110)
parser.add_argument('--penalty-after-words', '-penwords', type=int, help='Maximum number of words per language, before a length penalty is applied', default=80)
parser.add_argument('--penalty-per-word', '-wordpen', type=float, help='Penalty applied for each word when maximum number of unpenalized words have been reached', default=0.01)
parser.add_argument('--anchoring-delimiter', '-anchor', type=int, help='Maximum nodes in the alignment graph, before applying hard delimiters.', default=4000000)
parser.add_argument('--maximum-length-gale-church', '-maxgc', type=float, help='Maximum number of sentences in file for Gale-Church alignment. If longer, only greedy alignment selection applied', default=10000)
# Other settings
parser.add_argument('--proc-device', '-device', help='cuda for gpu, cpu if you don''t have an NVIDIA graphics card', default='cuda')
parser.add_argument('--num-proc', '-proc', help='number of processors to allocate for the pathfinding calculations', default=8)
args = parser.parse_args()

corpus_folder = args.corpus_folder
output_folder = args.corpus_folder + '/' + args.output_folder
source_language_folder = corpus_folder + '/' + args.source_language
target_language_folder = corpus_folder + '/' + args.target_language
align_info_folder = corpus_folder + '/align_info/'
if not os.path.exists(corpus_folder + '/tmp/'):
    os.makedirs(corpus_folder + '/tmp/')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(align_info_folder):
    os.makedirs(align_info_folder)
temporary_folder = corpus_folder + '/tmp/' + ''.join(random.choices(string.ascii_uppercase, k=8))
os.mkdir(temporary_folder)
file_in_list = args.filename

max_concats = args.max_concatenations
max_galechurch = args.maximum_length_gale_church
score_cutoff = args.score_cutoff
free_concats = args.free_concatenations
minimum_length_words = args.minimum_length_words
maximum_length_words = args.maximum_length_words
start_penalty_word_number = args.penalty_after_words
penalty_per_word = args.penalty_per_word

max_length = 128
batch_size = 128
printline = ''

cutoff4anchoring = args.anchoring_delimiter
cutoff_penalty = 1.25
minimum_labse_anchor = 0.95
labse_subtraction = 0.05
abs_minimum_labse_anchor = 0.8

def print_progress(infoclass):
    infoclass.update_times()
    print('{0: <26}'.format(infoclass.file_processing_stage) + '{0: <25}'.format(infoclass.input_file) + '{0: <18}'.format(str(infoclass.source_file_length) + ":" + str(infoclass.target_file_length)), '{0: <18}'.format("Files left: " + str(infoclass.files_left)),
          end='\r')

def load_labse_model(proc_device, labse_model):
    print("Loading LaBSE model...", end='\r')
    tokenizer = BertTokenizerFast.from_pretrained(labse_model)
    model = BertModel.from_pretrained(labse_model)
    model = model.eval()
    model.to(proc_device)
    return tokenizer, model


##### OVERLAP FUNCTIONS #####
##### from vecalign #####
def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line


def layer(lines, num_overlaps, comb=' '):
    """
    make front-padded overlapping sentences
    """
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        temp_list = [str(linenum) for linenum in range(ii, ii + num_overlaps)]
        out.append([comb.join(lines[ii:ii + num_overlaps]), ','.join(temp_list)])
    return out


def yield_overlaps(lines, num_overlaps):
    lines = [preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for layer_out in layer(lines, overlap):
            try:
                out_line, out_line_numbers = layer_out
                # check must be here so all outputs are unique
                out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
                yield out_line2, out_line_numbers
            except:
                pass


def get_overlaps(output_file, input_file, num_overlaps):
    output = []
    output_numbers = []
    output_overlaps = []
    output_lengthok = []
    lines = open(input_file, 'rt', encoding="utf-8").readlines()
    for out_line, out_line_numbers in yield_overlaps(lines, num_overlaps):
        if (len(out_line.split()) >= minimum_length_words) and (len(out_line.split()) <= maximum_length_words):
            output_lengthok.append(True)
        else:
            output_lengthok.append(False)
        output.append(out_line)
        output_numbers.append(out_line_numbers)
        output_overlaps.append(len(out_line_numbers.split(',')))
    with open(output_file, 'wt', encoding="utf-8") as fout, open(output_file + '.linenumbers', 'wt', encoding="utf-8") as fout_lines, open(output_file + '.overlaps', 'wt', encoding="utf-8") as fout_overlaps, open(output_file + '.lengthok', 'wt', encoding="utf-8") as fout_lengthok:
        for o_line in output:
            fout.write(o_line + '\n')
        for on_line in output_numbers:
            fout_lines.write(on_line + '\n')
        for oo_line in output_overlaps:
            fout_overlaps.write(str(oo_line) + '\n')
        for ol_line in output_lengthok:
            fout_lengthok.write(str(ol_line) + '\n')

##### LABSE FUNCTIONS #####
def create_list(infile):
    file_in = open(infile, 'r')
    fileLines_in = file_in.readlines()

    with open(infile + '.used', 'w') as outfile:
        in_lists = []
        in_list = []
        ctr = 0
        for i in fileLines_in:
            temp = i.strip().lstrip()
            temp2 = re.sub('[_]+', '_', temp)
            outfile.write(i)
            in_list.append(temp2)

            ctr += 1
            if ctr == batch_size:
                ctr = 0
                in_lists.append(in_list)
                in_list = []
        if len(in_list) > 0:
            in_lists.append(in_list)
    return in_lists


def emb_file(in_list, in_numpy):
    ctr = 0
    for i in in_list:
        ctr += 1
        inputs = tokenizer(i, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens = True)
        inputs = inputs.to(args.proc_device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output
            normalized_embeddings = F.normalize(embeddings, p=2)
            nump = normalized_embeddings.cpu().detach().numpy()
            if in_numpy is None:
                in_numpy = nump
            else:
                in_numpy = np.concatenate((in_numpy, nump), axis=0)
    return in_numpy


def create_emb_file(infile, maximum_length):
    temp_max_length = maximum_length
    embs_numpy = None
    while temp_max_length > 30: #arbitrary number, allow user to select in arguments?
        try:
            sentence_list = create_list(infile)
            embs_numpy = emb_file(sentence_list, embs_numpy)
            break
        except Exception as e:
            print(e, temp_max_length)
            temp_max_length = temp_max_length * 0.75 #arbitrary number, allow user to select in arguments?
    f_out = open(infile + '.labse_emb', 'wb')
    np.save(f_out, embs_numpy)
    f_out.close()
    torch.cuda.empty_cache()


##### ALIGNMENT FUNCTIONS #####
def read_sentences(file_folder, file_name):
    sentence_dict = {}
    ctr = 0
    with open(file_folder + '/' + file_name, 'r') as f:
        for sentence in f:
            sentence_dict[ctr] = sentence.strip()
            ctr += 1
    return sentence_dict


def open_emb_file(file_name):
    emb_dict = {}
    emb_file = open(file_name + '.used')
    emb_lines = emb_file.readlines()
    embs = np.load(file_name + '.labse_emb', allow_pickle=True)
    for i in range(0, len(emb_lines)):
        emb_dict[emb_lines[i].strip()] = embs[i]
    return emb_dict


def get_pairs(path, source_dict, target_dict, total_score):
    pairs = []
    path_pairs = path.strip().split('\n')
    score_list = total_score.strip().split('\n')
    ctr = 0
    for p in path_pairs:
        s,t = p.strip().replace('[','').replace(']','').split(':')
        s_nums = s.split(',')
        t_nums = t.split(',')
        source_out = ''
        target_out = ''
        if '' not in s_nums:
            for s_num in s_nums:
                source_out += ' ' + source_dict[int(s_num)]
            source_out = source_out.strip().lstrip()
        else:
            source_out = 'NULLALIGN'
        if '' not in t_nums:
            for t_num in t_nums:
                target_out += ' ' + target_dict[int(t_num)]
            target_out = target_out.strip().lstrip()
        else:
            target_out = 'NULLALIGN'
        pairs.append([source_out, target_out, score_list[ctr]])
        ctr += 1
    return pairs


def write_path_to_file(path, file_name, total_score):
    path = path.replace(':', ']:[').strip()
    out_path = ''
    pathlist = path.split('\n')
    scorelist = total_score.split('\n')
    ctr = 0
    for i in pathlist:
        out_path += i + ':' + scorelist[ctr] + '\n'
        ctr += 1
    with open(output_folder + '/' + file_name + '.path', 'w') as fo:
        fo.write(out_path)


def write_pairs_to_file(pairs, file_name, total_score):
    text_scores = []
    for i in total_score:
        if i != 0:
            text_scores.append(str(i))

    with open(output_folder + '/' + file_name + '.aligned', 'w') as fo:
        ctr = 0
        for p in pairs:
            outstring = p[0] + '\t' + p[1] + '\t' + p[2] + '\n'
            ctr += 1
            if outstring.find('NULLALIGN') == -1:
                fo.write(outstring)


######### MAIN #########
#### for calculating labse for everything
def similarity_matrix(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    matrix_multi = torch.matmul(normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1))
    return matrix_multi


def score_labse_matrix(source_list, target_list):
    source_inputs = tokenizer(source_list, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True)
    source_inputs = source_inputs.to(args.proc_device)
    target_inputs = tokenizer(target_list, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True)
    target_inputs = target_inputs.to(args.proc_device)

    with torch.no_grad():
        source_outputs = model(**source_inputs)
        target_outputs = model(**target_inputs)

    source_embeddings = source_outputs.pooler_output
    target_embeddings = target_outputs.pooler_output

    score = similarity_matrix(source_embeddings, target_embeddings)
    return score


def score_labse_matrix_split(anchor_source_list, anchor_target_list, anchor_source_dict, anchor_target_dict):
    out_matrix = np.array([[0 for x in range(len(anchor_target_list))] for y in range(len(anchor_source_list))])
    i_ctr = 0
    for i in anchor_source_list:
        j_ctr = 0
        for j in anchor_target_list:
            try:
                labse_score = anchor_target_dict[j.strip()].dot(anchor_source_dict[i.strip()].transpose())
                out_matrix[i_ctr][j_ctr] = labse_score
                j_ctr += 1
            except Exception as e:
                print(e)
                print("error in score_labse_matrix_split")
                print(i)
                print(j)

        i_ctr += 1
    return out_matrix


def galechurchSupport(file_name):
    alignments_out, highest_score, len_x, len_y = gale_church(source_language_folder + '/' + file_name, target_language_folder + '/' + file_name, 5000, None) # allow setting max_length for galechurch
    galechurch_alignments = []

    for a in alignments_out.split('\n'):
        try:
            gale_current = a.replace('[', '').replace(']', '').replace(' ', '').split(':')
            gale_source_lines, gale_target_lines = gale_current
            if (len(gale_source_lines.strip()) > 0 and len(gale_target_lines.strip()) > 0):
                galechurch_alignments.append([gale_source_lines, gale_target_lines])
        except Exception as e:
            pass
    return galechurch_alignments


def anchorsLoop(cutoff4anchoring, galechurch_alignments, src_emb_dict, trg_emb_dict, minimum_anchor_score, anchor_source_list_lines,anchor_target_list_lines, source_len, target_len, anchor_source_list, anchor_target_list, start_source, start_target, anchor_in, source_loc_start_list, source_loc_end_list, target_loc_start_list, target_loc_end_list, labse_score_matrix):
    anchor_list = [anchor_in]
    try:
        end_source = int(anchor_in[0].split(',')[0])
    except:
        end_source = anchor_in[0]
    try:
        end_target = int(anchor_in[1].split(',')[0])
    except:
        end_target = anchor_in[1]

    try:
        if len(galechurch_alignments) > 0:
            processInfo.set_status("Calculating anchors without matrix")
            anchor_list = calculate_anchor_nomatrix_set(galechurch_alignments, src_emb_dict, trg_emb_dict, minimum_anchor_score,
                                                        anchor_source_list_lines, anchor_target_list_lines, source_len, target_len,
                                                        anchor_source_list, anchor_target_list, start_source, end_source,
                                                        start_target, end_target, labse_score_matrix,
                                                        source_loc_start_list, source_loc_end_list, target_loc_start_list, target_loc_end_list)
    except:
        print("anchorsLoop failed")

    anchors_out = []
    maxNumberOfKnots = 0

    if minimum_anchor_score > abs_minimum_labse_anchor:
        for anchor in anchor_list:
            processInfo.set_status('anchorsLoop ' + f'{minimum_anchor_score:.2f}')
            try:
                end_source = int(anchor[0].split(',')[0])
            except:
                end_source = anchor[0]
            try:
                end_target = int(anchor[1].split(',')[0])
            except:
                end_target = anchor[1]
            currentNumberOfKnots = max(maxNumberOfKnots, (end_source - start_source) * (end_target - start_target))
            if currentNumberOfKnots > cutoff4anchoring:
                try:
                    new_anchor_list = anchorsLoop(cutoff4anchoring, galechurch_alignments, src_emb_dict, trg_emb_dict,
                                              minimum_anchor_score - 0.02, anchor_source_list_lines,
                                              anchor_target_list_lines, source_len, target_len, anchor_source_list,
                                              anchor_target_list, start_source, start_target, anchor, source_loc_start_list, source_loc_end_list, target_loc_start_list, target_loc_end_list, labse_score_matrix)
                    for nal in new_anchor_list:
                        if nal not in anchors_out:
                            anchors_out.append(nal)
                except Exception as e:
                    print(e)
                    print("anchorsLoop failed")

            else:
                if anchor not in anchors_out:
                    anchors_out.append(anchor)
            try:
                start_source = int(anchor[0].split(',')[0])
            except:
                start_source = anchor[0]
            try:
                start_target = int(anchor[1].split(',')[0])
            except:
                start_target = anchor[1]
    return anchors_out


def create_anchor_files(file_name, sentence_overlap, lengthfilter):
    src_overlaps_file = temporary_folder + '/overlaps.' + file_name + '.src'
    trg_overlaps_file = temporary_folder + '/overlaps.' + file_name + '.trg'
    src_overlaps_file_used = temporary_folder + '/overlaps.' + file_name + '.src.used'
    trg_overlaps_file_used = temporary_folder + '/overlaps.' + file_name + '.trg.used'
    src_embeddings_file = src_overlaps_file + '.labse_emb'
    trg_embeddings_file = trg_overlaps_file + '.labse_emb'
    anchor_overlaps_source_file = temporary_folder + '/anchoroverlaps.' + file_name + '.src.'
    anchor_overlaps_target_file = temporary_folder + '/anchoroverlaps.' + file_name + '.trg.'

    src_num_overlaps_file = open(src_overlaps_file + '.overlaps').readlines()
    src_length_file = open(src_overlaps_file + '.lengthok').readlines()
    src_linunum_file = open(src_overlaps_file + '.linenumbers').readlines()
    src_overlaps_used = open(src_overlaps_file_used).readlines()
    trg_num_overlaps_file = open(trg_overlaps_file + '.overlaps').readlines()
    trg_length_file = open(trg_overlaps_file + '.lengthok').readlines()
    trg_linunum_file = open(trg_overlaps_file + '.linenumbers').readlines()
    trg_overlaps_used = open(trg_overlaps_file_used).readlines()

    anchor_source_list = []
    anchor_target_list = []
    anchor_source_list_lines = []
    anchor_target_list_lines = []
    src_emb_dict = {}
    trg_emb_dict = {}
    with open(anchor_overlaps_source_file + str(sentence_overlap) + '.used', 'w') as anchor_src_overlaps_file, open(anchor_overlaps_target_file + str(sentence_overlap) + '.used', 'w') as anchor_trg_overlaps_file, open(anchor_overlaps_source_file + str(sentence_overlap) + '.linenumbers', 'w') as anchor_src_overlaps_file_linenumbers, open(anchor_overlaps_target_file + str(sentence_overlap) + '.linenumbers', 'w') as anchor_trg_overlaps_file_linenumbers:
        embs = np.load(src_overlaps_file + '.labse_emb', allow_pickle=True)
        for line_ctr in range(0, len(src_overlaps_used)):
            num_overlaps = int(src_num_overlaps_file[line_ctr])
            length_ok = bool(src_length_file[line_ctr])
            # anchors are only generated for one or two overlaps
            if sentence_overlap >= num_overlaps:
                if length_ok and lengthfilter:
                    anchor_src_overlaps_file.write(src_overlaps_used[line_ctr])
                    anchor_source_list.append(src_overlaps_used[line_ctr])
                    anchor_src_overlaps_file_linenumbers.write(src_linunum_file[line_ctr])
                    anchor_source_list_lines.append(src_linunum_file[line_ctr])
                    src_emb_dict[src_overlaps_used[line_ctr].strip()] = embs[line_ctr]
                else:
                    if not lengthfilter:
                        anchor_src_overlaps_file.write(src_overlaps_used[line_ctr])
                        anchor_source_list.append(src_overlaps_used[line_ctr])
                        anchor_src_overlaps_file_linenumbers.write(src_linunum_file[line_ctr])
                        anchor_source_list_lines.append(src_linunum_file[line_ctr])
                        src_emb_dict[src_overlaps_used[line_ctr].strip()] = embs[line_ctr]

        embs = np.load(trg_overlaps_file + '.labse_emb', allow_pickle=True)
        curr_emb_dict = {}
        for line_ctr in range(0, len(trg_overlaps_used)):
            num_overlaps = int(trg_num_overlaps_file[line_ctr])
            length_ok = bool(trg_length_file[line_ctr])
            if sentence_overlap >= num_overlaps:
                if length_ok and lengthfilter:
                    anchor_trg_overlaps_file.write(trg_overlaps_used[line_ctr])
                    anchor_target_list.append(trg_overlaps_used[line_ctr])
                    anchor_trg_overlaps_file_linenumbers.write(trg_linunum_file[line_ctr])
                    anchor_target_list_lines.append(trg_linunum_file[line_ctr])
                    trg_emb_dict[trg_overlaps_used[line_ctr].strip()] = embs[line_ctr]
                else:
                    if not lengthfilter:
                        anchor_trg_overlaps_file.write(trg_overlaps_used[line_ctr])
                        anchor_target_list.append(trg_overlaps_used[line_ctr])
                        anchor_trg_overlaps_file_linenumbers.write(trg_linunum_file[line_ctr])
                        anchor_target_list_lines.append(trg_linunum_file[line_ctr])
                        trg_emb_dict[trg_overlaps_used[line_ctr].strip()] = embs[line_ctr]
    # return overlaps as specified, and less
    # apply length filter as specified
    return anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, src_emb_dict, trg_emb_dict


def greedy_procedure_large(file_name, anchor_list, file_minimum_anchor_score, anchor_score_subtraction, temp_cutoff4anchoring, cutoffpenalty):
    processInfo.set_status('Greedy Anchoring')
    torch.cuda.empty_cache()
    maxNumberOfKnots = 9223372036854775807
    anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict = create_anchor_files(file_name, 1, True)

    while maxNumberOfKnots > temp_cutoff4anchoring:
        start_source = 0
        start_target = 0
        new_anchor_list = []
        maxNumberOfKnots = 0
        for anchor in anchor_list:
            try:
                end_source = int(anchor[0].split(',')[0])
            except:
                end_source = anchor[0]
            try:
                end_target = int(anchor[1].split(',')[0])
            except:
                end_target = anchor[1]

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
            if numberOfKnots > temp_cutoff4anchoring: #(30k x 30k matrix or equivalent)
                temp_greedy_anchor = greedy_anchor_selection_large(file_minimum_anchor_score, labse_score_matrix)
                if temp_greedy_anchor is not None:
                    greedy_anchor = [str(temp_greedy_anchor[0] + start_source), str(temp_greedy_anchor[1] + start_target)]
                    if greedy_anchor not in new_anchor_list:
                        new_anchor_list.append(greedy_anchor)
            if anchor not in new_anchor_list:
                new_anchor_list.append(anchor)
            try:
                start_source = int(anchor[0].split(',')[0])
            except:
                start_source = int(anchor[0])
            try:
                start_target = int(anchor[1].split(',')[0])
            except:
                start_target = int(anchor[1])
        anchor_list = new_anchor_list.copy()
        temp_cutoff4anchoring = temp_cutoff4anchoring * cutoffpenalty
        file_minimum_anchor_score = file_minimum_anchor_score - anchor_score_subtraction
    labse_score_matrix = [1]
    for i in labse_score_matrix:
        pass
    return anchor_list


def greedy_procedure(file_name, anchor_list, file_minimum_anchor_score, anchor_score_subtraction, temp_cutoff4anchoring, cutoffpenalty):
    anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict = create_anchor_files(file_name, 1, True)
    # this will not work if we have overlaps (other than 1)
    labse_score_matrix = np.asarray(create_labse_score_matrix(anchor_source_list, anchor_target_list, anchor_src_emb_dict, anchor_trg_emb_dict))

    torch.cuda.empty_cache()
    maxNumberOfKnots = 9223372036854775807
    while maxNumberOfKnots > temp_cutoff4anchoring:
        start_source = 0
        start_target = 0
        new_anchor_list = []
        for anchor in anchor_list:
            try:
                end_source = int(anchor[0].split(',')[0])
            except:
                end_source = anchor[0]
            try:
                end_target = int(anchor[1].split(',')[0])
            except:
                end_target = anchor[1]
            numberOfKnots = (end_source - start_source) * (end_target - start_target)
            if numberOfKnots > temp_cutoff4anchoring:
                greedy_anchor = greedy_anchor_selection(start_source, start_target, anchor,
                                                        anchor_source_list_lines,
                                                        anchor_target_list_lines,
                                                        file_minimum_anchor_score,
                                                        labse_score_matrix)
                if greedy_anchor is not None:
                    if greedy_anchor not in new_anchor_list:
                        new_anchor_list.append(greedy_anchor)
            if anchor not in new_anchor_list:
                new_anchor_list.append(anchor)
            try:
                start_source = int(anchor[0].split(',')[0])
            except:
                start_source = int(anchor[0])
            try:
                start_target = int(anchor[1].split(',')[0])
            except:
                start_target = int(anchor[1])
        anchor_list = new_anchor_list.copy()
        temp_cutoff4anchoring = temp_cutoff4anchoring * cutoffpenalty
        file_minimum_anchor_score = file_minimum_anchor_score - anchor_score_subtraction
    labse_score_matrix = [1]
    for i in labse_score_matrix:
        pass
    return anchor_list

def create_anchors_nomatrix(file_name, file_minimum_labse_anchor, source_len, target_len, galechurch_alignments):
    # This function works if there are very many knots (tens of millions) so that a matrix of scores can't be loaded into memory
    # It is slower than the matrix version, but it works for large corpora
    processInfo.set_anchoring()
    processInfo.set_status('create_anchors_nomatrix')
    torch.cuda.empty_cache()

    anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, src_emb_dict, trg_emb_dict = create_anchor_files(file_name, 2, False)

    start_source = 0
    start_target = 0
    end_source = source_len
    end_target = target_len

    anchor_list = [[end_source, end_target]]
    maxNumberOfKnots = source_len * target_len
    if maxNumberOfKnots > cutoff4anchoring:
        if len(galechurch_alignments) > 0:
            source_loc_start_list, source_loc_end_list = loc_start_end_matrices(anchor_source_list_lines)
            target_loc_start_list, target_loc_end_list = loc_start_end_matrices(anchor_target_list_lines)
            labse_score_matrix = np.asarray(create_labse_score_matrix(anchor_source_list, anchor_target_list, src_emb_dict, trg_emb_dict))

            anchor_list = anchorsLoop(cutoff4anchoring, galechurch_alignments, src_emb_dict, trg_emb_dict, file_minimum_labse_anchor,
                                      anchor_source_list_lines, anchor_target_list_lines, source_len, target_len, anchor_source_list,
                                      anchor_target_list, start_source, start_target, anchor_list[0],
                                      source_loc_start_list, source_loc_end_list, target_loc_start_list, target_loc_end_list, labse_score_matrix)

    start_source = 0
    start_target = 0
    greedy_flag = False
    for anchor in anchor_list:
        try:
            end_source = int(anchor[0].split(',')[0])
        except:
            end_source = anchor[0]

        try:
            end_target = int(anchor[1].split(',')[0])
        except:
            end_target = anchor[1]
        numberOfKnots = (end_source - start_source) * (end_target - start_target)
        if numberOfKnots > cutoff4anchoring:
            greedy_flag = True
        try:
            start_source = int(anchor[0].split(',')[0])
        except:
            start_source = anchor[0]
        try:
            start_target = int(anchor[1].split(',')[0])
        except:
            start_target = anchor[1]

    if greedy_flag:
        processInfo.set_status('Greedy Anchoring')
        start_greedy = time.process_time()
        anchor_list = greedy_procedure(file_name, anchor_list, file_minimum_labse_anchor, labse_subtraction, cutoff4anchoring, cutoff_penalty)
        end_greedy = time.process_time()
        elapsed_greedy = end_greedy - start_greedy
        processInfo.set_elapsed_greedy(elapsed_greedy)
    return anchor_list


def process_file(filename, minimum_labse_anchor):
    file_name = filename.split('/')[-1]
    processInfo.init_file(file_name)
    file_minimum_labse_anchor = minimum_labse_anchor

    ## Read file dictionaries ##
    source_dict = read_sentences(source_language_folder, file_name)
    target_dict = read_sentences(target_language_folder, file_name)
    source_len = len(source_dict.keys())
    target_len = len(target_dict.keys())
    anchor_list = [[str(source_len), str(target_len)]]

    processInfo.set_file(source_len, target_len)
    maxNumberOfKnots = source_len * target_len

    ## overlaps ##
    #source
    src_input_file = source_language_folder + '/' + file_name
    src_overlaps_file = temporary_folder + '/overlaps.' + file_name + '.src'
    get_overlaps(src_overlaps_file, src_input_file, max_concats)
    #target
    trg_input_file = target_language_folder + '/' + file_name
    trg_overlaps_file = temporary_folder + '/overlaps.' + file_name + '.trg'
    get_overlaps(trg_overlaps_file, trg_input_file, max_concats)

    ## create LaBSE embeddings ##
    create_emb_file(temporary_folder + '/overlaps.' + file_name + '.src', max_length)
    create_emb_file(temporary_folder + '/overlaps.' + file_name + '.trg', max_length)

    torch.cuda.empty_cache()
    if source_len > 0 and target_len > 0:
        if (source_len > max_galechurch) or (target_len > max_galechurch):
            ## File large - straight to greedy anchoring ##
            processInfo.set_status("Greedy anchoring large file...")
            start_greedy = time.process_time()
            max_matrix_size_for_greedy = 1000000000
            if (source_len*target_len) > max_matrix_size_for_greedy:
                slow_greedy = True
                anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict = create_anchor_files(file_name, 1, True)

                while slow_greedy:
                    new_anchor_list = []
                    slow_greedy = False
                    start_source = 0
                    start_target = 0
                    for anchor in anchor_list:
                        if (int(anchor[0])-int(start_source))*(int(anchor[1])-int(start_target)) > max_matrix_size_for_greedy:
                            new_anchor = get_highest_labse_anchor(start_source, start_target, anchor, anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict)
                            if new_anchor is not None:
                                new_anchor_list.append(new_anchor)
                                try:
                                    new_matrix_first = (int(new_anchor[0]) - start_source) * (int(new_anchor[1]) - start_target)
                                    new_matrix_second = (int(anchor[0]) - int(new_anchor[0])) * (int(anchor[1]) - int(new_anchor[1]))
                                    if (new_matrix_first > max_matrix_size_for_greedy) or (new_matrix_second > max_matrix_size_for_greedy):
                                        slow_greedy = True
                                except:
                                    pass
                            else:
                                slow_greedy = True
                        new_anchor_list.append(anchor)
                        start_source = int(anchor[0])
                        start_target = int(anchor[1])
                    anchor_list = new_anchor_list.copy()
                    print('slow greedy anchor list:', anchor_list)

            file_minimum_labse_anchor = minimum_labse_anchor
            anchor_list = greedy_procedure_large(file_name, anchor_list, file_minimum_labse_anchor, labse_subtraction, cutoff4anchoring, cutoff_penalty)
            end_greedy = time.process_time()
            elapsed_greedy = end_greedy - start_greedy
            processInfo.set_elapsed_greedy(elapsed_greedy)
        else:
            if maxNumberOfKnots > cutoff4anchoring:
                processInfo.set_status("Calculating anchors...")
                anchor_source_list, anchor_target_list, anchor_source_list_lines, anchor_target_list_lines, anchor_src_emb_dict, anchor_trg_emb_dict = create_anchor_files(file_name, 2, True)

                try:
                    processInfo.set_status("Doing Gale-Church...")
                    start_galechurch = time.process_time()
                    galechurch_alignments = galechurchSupport(file_name)
                    end_galechurch = time.process_time()
                    elapsed_galechurch = end_galechurch - start_galechurch
                    processInfo.set_elapsed_gale_church(elapsed_galechurch)

                    if len(galechurch_alignments) > 0:
                        try:
                            start_calclabse = time.process_time()
                            processInfo.set_status("Calculating LaBSE matrix")
                            try:
                                # fastest approach - create labse matrix. But if it gets too large it will explode.
                                labse_matrix = score_labse_matrix(anchor_source_list, anchor_target_list).cpu().numpy()
                                torch.cuda.empty_cache()
                            except Exception as e:
                                # slower approach - calculate each knot in the matrix individually. Doesn't use as much memory while calculating.
                                torch.cuda.empty_cache()
                                processInfo.set_anchoring()
                                labse_matrix = score_labse_matrix_split(anchor_source_list, anchor_target_list, anchor_src_emb_dict, anchor_trg_emb_dict)
                                torch.cuda.empty_cache()

                            source_start_matrix = []
                            source_end_matrix = []
                            target_start_matrix = []
                            target_end_matrix = []
                            end_calclabse = time.process_time()
                            elapsed_calclabse = end_calclabse - start_calclabse
                            processInfo.set_elapsed_calc_labse(elapsed_calclabse)

                            processInfo.set_status("Creating start/end matrices")
                            for i in range(0, len(anchor_source_list_lines)):
                                source_start_matrix.append(int(anchor_source_list_lines[i].strip().split(',')[0]))
                                source_end_matrix.append(int(anchor_source_list_lines[i].strip().split(',')[-1]))

                            for j in range(0, len(anchor_target_list_lines)):
                                target_start_matrix.append(int(anchor_target_list_lines[j].strip().split(',')[0]))
                                target_end_matrix.append(int(anchor_target_list_lines[j].strip().split(',')[-1]))

                            start_calcanchors = time.process_time()
                            processInfo.set_status("Calc anchors... " + str(file_minimum_labse_anchor))
                            start_source = 0
                            start_target = 0
                            anchor_list = calculate_anchor_set(source_start_matrix, source_end_matrix, target_start_matrix, target_end_matrix, galechurch_alignments, labse_matrix, file_minimum_labse_anchor, anchor_source_list_lines, anchor_target_list_lines, start_source, source_len, start_target, target_len)

                            while maxNumberOfKnots > cutoff4anchoring:
                                processInfo.set_status("Calc anchors... " + str(file_minimum_labse_anchor))
                                file_minimum_labse_anchor -= labse_subtraction
                                new_anchor_list = []
                                maxNumberOfKnots = 0
                                start_source = 0
                                start_target = 0
                                for anchor in anchor_list:
                                    try:
                                        end_source = int(anchor[0].split(',')[0])
                                    except:
                                        end_source = int(anchor[0])

                                    try:
                                        end_target = int(anchor[1].split(',')[0])
                                    except:
                                        end_target = int(anchor[1])

                                    numberOfKnots = (end_source - start_source) * (end_target - start_target)
                                    maxNumberOfKnots = max(maxNumberOfKnots, numberOfKnots)
                                    if numberOfKnots > cutoff4anchoring:
                                        temp_anchor_list = calculate_anchor_set(source_start_matrix, source_end_matrix, target_start_matrix, target_end_matrix, galechurch_alignments, labse_matrix,
                                                                           file_minimum_labse_anchor, anchor_source_list_lines,
                                                                           anchor_target_list_lines,
                                                                           start_source, end_source, start_target, end_target)
                                        for temp_anchor in temp_anchor_list:
                                            if temp_anchor not in new_anchor_list:
                                                new_anchor_list.append(temp_anchor)

                                    if anchor not in new_anchor_list:
                                        new_anchor_list.append(anchor)
                                    try:
                                        start_source = int(anchor[0].split(',')[0])
                                    except:
                                        start_source = int(anchor[0])
                                    try:
                                        start_target = int(anchor[1].split(',')[0])
                                    except:
                                        start_target = int(anchor[1])
                                anchor_list = new_anchor_list.copy()
                                if file_minimum_labse_anchor < abs_minimum_labse_anchor:
                                    break
                            end_calcanchors = time.process_time()
                            elapsed_calcanchors = end_calcanchors - start_calcanchors
                            processInfo.set_elapsed_calc_anchors(elapsed_calcanchors)
                            processInfo.set_status("Greedy anchoring...")
                            start_greedy = time.process_time()
                            anchor_list = greedy_procedure(file_name, anchor_list, minimum_labse_anchor, labse_subtraction, cutoff4anchoring, cutoff_penalty)
                            end_greedy = time.process_time()
                            elapsed_greedy = end_greedy - start_greedy
                            processInfo.set_elapsed_greedy(elapsed_greedy)
                        except Exception as e:
                            print('Out of memory... Trying another approach... This may take some time...')
                            print(e)
                            source_len = len(source_dict.keys())
                            target_len = len(target_dict.keys())
                            anchor_list = create_anchors_nomatrix(file_name, file_minimum_labse_anchor-labse_subtraction, source_len, target_len, galechurch_alignments)
                    else:
                        processInfo.set_status("Greedy anchoring...")
                        start_greedy = time.process_time()
                        anchor_list = greedy_procedure(file_name, anchor_list, file_minimum_labse_anchor, labse_subtraction, cutoff4anchoring, cutoff_penalty)
                        end_greedy = time.process_time()
                        elapsed_greedy = end_greedy - start_greedy
                        processInfo.set_elapsed_greedy(elapsed_greedy)
                except:
                    anchor_list = [[source_len, target_len]]
            else:
                anchor_list = [[source_len, target_len]]

        if anchor_list == []:
            anchor_list = [[source_len, target_len]]

        processInfo.set_anchors(anchor_list)

        ## Read LaBSE embeddings ##
        processInfo.set_status("Reading LaBSE embeddings")
        src_emb_dict = open_emb_file(temporary_folder + '/overlaps.' + file_name + '.src')
        trg_emb_dict = open_emb_file(temporary_folder + '/overlaps.' + file_name + '.trg')

        ## Run aligner ##
        processInfo.set_status("Creating anchor chunks")
        start_anchor = (0, 0)
        matrix_anchors = ()
        for curr_anchor in anchor_list:
            try:
                source_end_anchor = int(curr_anchor[0].split(',')[0])
            except:
                source_end_anchor = int(curr_anchor[0])
            try:
                target_end_anchor = int(curr_anchor[1].split(',')[0])
            except:
                target_end_anchor = int(curr_anchor[1])
            this_anchor = (source_end_anchor, target_end_anchor)
            matrix_anchors += (((start_anchor[0]-1, start_anchor[1]-1), (this_anchor[0],this_anchor[1])),)
            start_anchor = this_anchor

        torch.cuda.empty_cache()
        processInfo.set_status("Aligning...")
        start_align = time.process_time()

        total_path, total_score = align_anchors_multi(matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict, args.num_proc, score_cutoff,
                                         max_concats, processInfo, minimum_length_words, maximum_length_words, start_penalty_word_number,
                                         penalty_per_word, free_concats)

        finalpairs = get_pairs(total_path, source_dict, target_dict, total_score)
        align_info = processInfo.print_info()

        with open(align_info_folder + file_name + '.info', 'w') as fo:
            fo.write(align_info)

        write_path_to_file(total_path, file_name, total_score)
        write_pairs_to_file(finalpairs, file_name, total_score)
        end_align = time.process_time()
        align_elapsed = end_align - start_align
        processInfo.set_elapsed_align(align_elapsed)
    tempfiles = glob.glob(temporary_folder + '/*')
    for files2del in tempfiles:
        os.remove(files2del)

def get_filesleft(alignlist):
    filesdonefile = open(corpus_folder + '/filesdone.txt', 'r')
    searchset = filesdonefile.readlines()
    filesdonefile.close()
    donelist = []

    for f in searchset:
        try:
            donelist.append(f.strip())
        except:
            pass

    filesleft = list(set(alignlist) - set(donelist))
    return filesleft

# Create overlaps
if __name__ == '__main__':
    main_starttime = datetime.datetime.now()
    tokenizer, model = load_labse_model(args.proc_device, 'setu4993/LaBSE')
    files2align = {}
    filesdone = {}
    with open(corpus_folder + '/files2align.txt', 'r') as fa, open(corpus_folder + '/filesdone.txt', 'r') as fd:
        for line in fd:
            filesdone[line.strip()] = 1
        for line in fa:
            if line.strip() not in filesdone:
                files2align[line.strip()] = 1
    alignlist = list(files2align.keys())
    random.shuffle(alignlist)
    processInfo = ReportInfo(datetime.datetime.now(), len(alignlist))
    rt = RepeatedTimer(0.3, print_progress, processInfo)

    #if running multiple instances of the script on the same directory, this remakes the list of files to align
    redolist_interval = int(len(alignlist)/500)+1

    try:
        while len(alignlist) > 0:
            redolist_interval -= 1
            if redolist_interval < 1:
                processInfo.set_status('Getting files to align...')
                alignlist = get_filesleft(alignlist)
                processInfo.files_left = len(alignlist)
                redolist_interval = int(len(alignlist)/500)+1 #Allow the user to set this number (500) in the input parameters
            if len(alignlist) > 0:
                file = alignlist.pop()

                filesdonefile = open(corpus_folder + '/filesdone.txt', 'a')
                filesdonefile.write(file + '\n')
                filesdonefile.close()

                processInfo.files_left = len(alignlist)
                process_file(file, minimum_labse_anchor)
    finally:
        rt.stop()

    try:
        os.remove(temporary_folder + '/*')
    except:
        pass
    try:
        os.rmdir(temporary_folder)
    except:
        pass

    print('\nAligned ' + str(len(list(files2align.keys()))) + ' files in ' + str((datetime.datetime.now() - main_starttime).total_seconds()) + ' seconds.', flush=True)

