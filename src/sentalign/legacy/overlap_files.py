"""Version-1 overlap files and LaBSE encoding through temporary files (reference implementation).

Not used by the aligner any more (it was replaced by ``sentalign.document`` and ``sentalign.scoring``); kept so
that the tests can check that the new code produces exactly what these functions produced.
"""
import logging
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

log = logging.getLogger('sentalign.embeddings')


class ModelLoadError(RuntimeError):
    """The LaBSE model or tokenizer could not be loaded or moved to the device."""


def load_labse_model(proc_device, labse_model):
    import os
    # was a module-level side effect in sentAlign.py
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.info('Loading LaBSE model %s on %s', labse_model, proc_device)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(labse_model)
        model = BertModel.from_pretrained(labse_model)
        model = model.eval()
        model.to(proc_device)
    except Exception as e:
        raise ModelLoadError('could not load %s on device %r: %s' % (labse_model, proc_device, e)) from e
    return tokenizer, model


def empty_cuda_cache(proc_device):
    """torch.cuda.empty_cache(), only when the work is done on a CUDA device."""
    if str(proc_device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()


##### OVERLAP FUNCTIONS #####
# The three functions below (preprocess_line, layer, yield_overlaps) are taken from Vecalign
# (https://github.com/thompsonb/vecalign, overlap.py and dp_utils.py), Copyright 2019 Brian Thompson, licensed under
# the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0); SentAlign 1.0 changed them to also
# return the line numbers of each span.
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
            if isinstance(layer_out, str):
                # the 'PAD' front padding of layer(): version 1 skipped it through a failed tuple unpacking
                continue
            out_line, out_line_numbers = layer_out
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2, out_line_numbers


def get_overlaps(output_file, input_file, num_overlaps, minimum_length_words, maximum_length_words):
    output = []
    output_numbers = []
    output_overlaps = []
    output_lengthok = []
    with open(input_file, 'rt', encoding="utf-8") as f:
        lines = f.readlines()
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
def create_list(infile, batch_size):
    with open(infile, 'r', encoding='utf-8') as file_in:
        fileLines_in = file_in.readlines()

    with open(infile + '.used', 'w', encoding='utf-8') as outfile:
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


def emb_file(in_list, in_numpy, tokenizer, model, proc_device):
    for i in in_list:
        inputs = tokenizer(i, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True)
        inputs = inputs.to(proc_device)
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


def create_emb_file(infile, maximum_length, tokenizer, model, proc_device, batch_size):
    # `temp_max_length` is never used for truncation; the loop only repeats the same work up to six
    # times when an error occurs (e.g. CUDA out of memory).  Kept as in 1.0.
    temp_max_length = maximum_length
    embs_numpy = None
    while temp_max_length > 30:  # arbitrary number, allow user to select in arguments?
        try:
            sentence_list = create_list(infile, batch_size)
            embs_numpy = emb_file(sentence_list, embs_numpy, tokenizer, model, proc_device)
            break
        except Exception as e:
            log.warning('encoding %s failed (%s: %s), retrying (max length %s)', infile, type(e).__name__, e, temp_max_length)
            temp_max_length = temp_max_length * 0.75  # arbitrary number, allow user to select in arguments?
    with open(infile + '.labse_emb', 'wb') as f_out:
        np.save(f_out, embs_numpy)
    empty_cuda_cache(proc_device)


def open_emb_file(file_name):
    emb_dict = {}
    with open(file_name + '.used', encoding='utf-8') as emb_file_:
        emb_lines = emb_file_.readlines()
    embs = np.load(file_name + '.labse_emb', allow_pickle=True)
    for i in range(0, len(emb_lines)):
        emb_dict[emb_lines[i].strip()] = embs[i]
    return emb_dict
