# -*- coding: UTF-8 -*-
import multiprocessing
from itertools import repeat, combinations
import time
import logging

logger = logging.getLogger('sentenceAligner')

def align_anchors_multi(matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict, num_proc, score_cutoff,
                        max_concats, processInfo, minimum_length_words, maximum_length_words, start_penalty_word_number,
                        penalty_per_word, free_concats):
    total_path = ''
    total_calculations = 0
    for anchor_pair in matrix_anchors:
        source_length = anchor_pair[1][0] - anchor_pair[0][0] + 1
        target_length = anchor_pair[1][1] - anchor_pair[0][1] + 1
        total_calculations += source_length * target_length
    calcs_done = 0
    processInfo.set_total_calculations(total_calculations)


    if num_proc == 0:
        run_proc = 1
    else:
        run_proc = int(num_proc)

    with multiprocessing.Pool(processes=run_proc) as pool:
        res = pool.starmap(align_anchors,
                           zip(matrix_anchors, repeat(source_dict), repeat(target_dict), repeat(src_emb_dict),
                               repeat(trg_emb_dict), repeat(score_cutoff), repeat(max_concats),
                               repeat(minimum_length_words), repeat(maximum_length_words), repeat(start_penalty_word_number),
                               repeat(penalty_per_word), repeat(free_concats), repeat(processInfo)))
        for result in res:
            total_path += result[1]
    return total_path



def align_anchors(anchor, source_dict, target_dict, src_emb_dict, trg_emb_dict, score_cutoff, max_concats,
                  minimum_length_words, maximum_length_words, start_penalty_word_number, penalty_per_word, free_concats, processInfo):
    start_source = anchor[0][0]
    start_target = anchor[0][1]
    end_source = anchor[1][0]
    end_target = anchor[1][1]
    source_length = end_source - start_source + 1
    target_length = end_target - start_target + 1

    #best_score_array = [[(y+x)*score_cutoff for y in range(end_target - start_target)] for x in range(end_source - start_source)]
    best_score_array = [[0 for y in range(end_target - start_target)] for x in range(end_source - start_source)]
    #print(best_score_array[0])
    #print(best_score_array[1])
    best_path_array = [['' for y in range(end_target - start_target)] for x in range(end_source - start_source)]
    best_parent_array = [[(-1,-1) for y in range(end_target - start_target)] for x in range(end_source - start_source)]

    # skoða hvort ég sé að skrá skorin vitlaust, af hverju er (0,0) = 0,4 og það kemur alignment í (0,1)? Það er eins og 1,1 alignmentið skráist ekki

    for i in range(start_source, end_source):
        source_concats, source_concats_count_dict = create_concats(start_source, i, max_concats, source_length, source_dict)
        for j in range(start_target, end_target):
            target_concats, target_concats_count_dict = create_concats(start_target, j, max_concats, target_length, target_dict)

            if i == start_source:
                if j == start_target:
                    previous_score = 0
                    best_score_array[i - start_source][j - start_target] = 0
                    max_score = 0
                else:
                    previous_score = best_score_array[i-start_source][j-1-start_target]
                    best_score_array[i - start_source][j - start_target] = previous_score + score_cutoff
                    best_parent_array[i - start_source][j - start_target] = (i - start_source, j - 1 - start_target)
                    best_path_array[i - start_source][j - start_target] = '[:' + str(j) + ']\n'
                    max_score = previous_score + score_cutoff
            else:
                if j == start_target:
                    previous_score = best_score_array[i-1-start_source][j-start_target]
                    best_score_array[i - start_source][j - start_target] = previous_score + score_cutoff
                    best_parent_array[i - start_source][j - start_target] = (i - 1 - start_source, j - start_target)
                    best_path_array[i - start_source][j - start_target] = '[' + str(i) + ':]\n'
                    max_score = previous_score + score_cutoff
                else:
                    previous_score = best_score_array[i-1-start_source][j-1-start_target]
                    left_score = best_score_array[i-start_source][j-1-start_target]
                    up_score = best_score_array[i-1-start_source][j-start_target]
                    if left_score < up_score:
                        best_score_array[i - start_source][j - start_target] = up_score + score_cutoff
                        #ATH. er ég að velja parent array rétt? Á þetta að vera öfugt?
                        best_parent_array[i - start_source][j - start_target] = (i - 1 - start_source, j - start_target)
                        best_path_array[i - start_source][j - start_target] = '[' + str(i) + ':]\n'
                        max_score = up_score + score_cutoff
                    else:
                        best_score_array[i - start_source][j - start_target] = left_score + score_cutoff
                        best_parent_array[i - start_source][j - start_target] = (i - start_source, j - 1 - start_target)
                        best_path_array[i - start_source][j - start_target] = '[:' + str(j) + ']\n'
                        max_score = left_score + score_cutoff

            #ATHUGA ALLT SAMAN - leggja saman til að fá bestu leið jafnóðum

            #print(i-1-start_source, j-1-start_target)
            #max_score = previous_score + score_cutoff
            #max_score = previous_score

            for m in source_concats:
                m_ctr = source_concats_count_dict[m]-1
                for n in target_concats:
                    n_ctr = target_concats_count_dict[n]-1
                    try:
                        labse_score = trg_emb_dict[n].dot(src_emb_dict[m].transpose())
                    except:
                        labse_score = 0

                    total_splits = m_ctr+n_ctr+2

                    # af hverju ekki bara að skoða alltaf punktinn, breyti > í >=
                    if (i-(m_ctr+1)) >= start_source and (j-(n_ctr+1)) >= start_target:
                        try:
                            # af hverju er +1 á eftir start source og start target? tek það út
                            best_node_score = best_score_array[i - (m_ctr+1) - (start_source)][j - (n_ctr+1) - (start_target)]
                        except:
                            best_node_score = 0
                    else:
                        best_node_score = 0

                    s_concat_length = m.count(' ') + 1
                    t_concat_length = n.count(' ') + 1
                    if (minimum_length_words <= s_concat_length <= maximum_length_words) and (minimum_length_words <= t_concat_length <= maximum_length_words) and (score_cutoff <= labse_score):
                        curr_penalty = 0
                        if s_concat_length > start_penalty_word_number:
                            curr_penalty += (s_concat_length - start_penalty_word_number) * penalty_per_word
                        if t_concat_length > start_penalty_word_number:
                            curr_penalty += (t_concat_length - start_penalty_word_number) * penalty_per_word
                        if total_splits - free_concats >= 0:
                            calculated_score = (total_splits - free_concats) + 1
                            curralign_score = (((labse_score - curr_penalty) ** calculated_score) * total_splits) + best_node_score
                        else:
                            curralign_score = ((labse_score - curr_penalty) * total_splits) + best_node_score
                    else:
                        curralign_score = 0

                    if curralign_score >= max_score:
                        # set hér alignmentið í path-listann
                        max_score = curralign_score
                        path_addition = []
                        max_path = ''
                        temp_s = ''
                        temp_t = ''
                        for s_coordinate in range(i-m_ctr, i+1):
                            temp_s += str(s_coordinate) + ','
                        for t_coordinate in range(j-n_ctr, j+1):
                            temp_t += str(t_coordinate) + ','
                        path_addition.append('[' + temp_s.strip(',') + ':' + temp_t.strip(',') + ']')

                        for k in path_addition:
                            max_path += str(k) + '\n'
                        #aftur er +1 í öllu þegar gildin eru sett - af hverju?
                        best_score_array[i - (start_source)][j - (start_target)] = curralign_score
                        best_parent_array[i - (start_source)][j - (start_target)] = (i - (start_source) - (m_ctr+1),j - (start_target) - (n_ctr+1))
                        best_path_array[i - (start_source)][j - (start_target)] = max_path

                    #þetta fyrir neðan á að vera fyrir null alignment - en leysi ég það ekki í byrjun?
                    #if best_score_array[i - (start_source + 1)][j - (start_target + 1)] == previous_score: # þetta er auðvitað alltaf rétt...
                    #    if j > 0:
                    #        best_score_array[i - (start_source + 1)][j - (start_target + 1)] += score_cutoff
                    #        best_parent_array[i - (start_source + 1)][j - (start_target + 1)] = (i - (start_source + 1), j - (start_target + 1) - 1)
                    #        best_path_array[i - (start_source + 1)][j - (start_target + 1)] = '[:' + str(j) + ']\n'
                    #    else:
                    #        best_score_array[i - (start_source + 1)][j - (start_target + 1)] += score_cutoff
                    #        best_parent_array[i - (start_source + 1)][j - (start_target + 1)] = (i - (start_source + 1) - 1, j - (start_target + 1))
                    #        best_path_array[i - (start_source + 1)][j - (start_target + 1)] = '[' + str(i) + ':]\n'
                    n_ctr+=1
                m_ctr+=1
            processInfo.add_nodes(end_target - start_target)

    goOn = True
    current_node = (end_source - (start_source + 1), end_target - (start_target + 1))
    #print(current_node)
    #print(len(best_path_array))
    #print(len(best_path_array[0]))
    path = ''
    while goOn:
        if current_node == anchor[0]:
            goOn = False
        path = best_path_array[current_node[0]][current_node[1]] + path
        current_node = best_parent_array[current_node[0]][current_node[1]]
        if current_node[0] == -1 or current_node[1] == -1:
            goOn = False

    #print('------------------------------------')
    # print(best_score_array[0])
    # print(best_parent_array[0])
    # print(best_path_array[0])
    # print(best_score_array[1])
    # print(best_parent_array[1])
    # print(best_path_array[1])
    # print(best_score_array[2])
    # print(best_parent_array[2])
    # print(best_path_array[2])
    # print(best_score_array[3])
    # print(best_parent_array[3])
    # print(best_path_array[3])
    #print(best_score_array[-1])
    #print(best_parent_array[-1])
    #print(best_path_array[-1])


    calculated_nodes = (end_source - start_source) * (end_target - start_target)
    #print('previous path', path)
    #print('------------------------------------')
    source_nulls, target_nulls, path = reevaluate_path(path, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff)
    source_nulls = list(map(int, source_nulls))
    target_nulls = list(map(int, target_nulls))

    #print('reevaluate_path', str(path))
    source_nulls, target_nulls, path = check_for_nulls(path, source_nulls, target_nulls)
    source_nulls = list(map(int, source_nulls))
    target_nulls = list(map(int, target_nulls))
    new_path = add_nulls(source_nulls, target_nulls, path, src_emb_dict, trg_emb_dict, source_dict, target_dict)
    ctr = 0
    while new_path != path:
        ctr += 1
        #print('adding nulls', str(ctr))
        path = new_path
        new_path = add_nulls(source_nulls, target_nulls, path, src_emb_dict, trg_emb_dict, source_dict, target_dict)

    path = fill_null_aligns(new_path, end_source, end_target)

    #print('------------------------------------')
    #print('Path: ', str(path.split('\n')))
    return calculated_nodes, path

def fill_null_aligns(new_path, end_source, end_target):
    path = ''
    start_source = -1
    start_target = -1
    for line in new_path.split('\n'):
        if len(line) > 0:
            curr_source, curr_target = line.split(':')
            curr_source = curr_source.strip('[]').split(',')
            if curr_source[0] != start_source + 1:
                for i in range(start_source + 1, int(curr_source[0])):
                    path += '[' + str(i) + ':]\n'
            curr_target = curr_target.strip('[]').split(',')
            if curr_target[0] != start_target + 1:
                for i in range(start_target + 1, int(curr_target[0])):
                    path += '[:' + str(i) + ']\n'
            start_source = int(curr_source[-1])
            start_target = int(curr_target[-1])
            path += line + '\n'
    if start_source != end_source:
        for i in range(start_source + 1, end_source):
            path += '[' + str(i) + ':]\n'
    if start_target != end_target:
        for i in range(start_target + 1, end_target):
            path += '[:' + str(i) + ']\n'
    return path


def check_for_nulls(path, source_nulls, target_nulls):
    splitpath = path.strip().split('\n')
    outpath = ''
    for i in splitpath:
        curr_path = i.strip('[]').split(':')
        if curr_path[0] == '':
            target_nulls.append(curr_path[1])
        elif curr_path[1] == '':
            source_nulls.append(curr_path[0])
        else:
            outpath += i + '\n'
    return source_nulls, target_nulls, outpath

def add_nulls(source_nulls, target_nulls, path, src_emb_dict, trg_emb_dict, source_dict, target_dict):
    splitpath = path.strip().split('\n')

    for i in splitpath:
        curr_path = i
        best_node = i
        try:
            current = curr_path.strip('[]').split(':')
            source = current[0].split(',')
            target = current[1].split(',')
            # get baseline labse score
            labse_value = get_labse_score(source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict)
            #if int(source[-1]) == 18:
            #    print('source', source)
            #    print('labse_value', labse_value)

            if (int(source[0])-1) in source_nulls:
                #print('source null', source[0])
                current_source = [str(int(source[0])-1)] + source
                current_labse_value = get_labse_score(current_source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict)
                if current_labse_value > labse_value:
                    labse_value = current_labse_value
                    best_node = '[' + ','.join(current_source) + ':' + ','.join(target) + ']'
            if (int(target[0])-1) in target_nulls:
                #print('target null', target[0])
                current_target = [str(int(target[0])-1)] + target
                current_labse_value = get_labse_score(source, current_target, src_emb_dict, trg_emb_dict, source_dict, target_dict)
                if current_labse_value > labse_value:
                    labse_value = current_labse_value
                    best_node = '[' + ','.join(source) + ':' + ','.join(current_target) + ']'
            if (int(source[-1])+1) in source_nulls:
                #print('source null', source[-1])
                current_source = source + [str(int(source[-1])+1)]
                current_labse_value = get_labse_score(current_source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict)
                #if int(source[-1]) == 18:
                #    print('current_source', current_source)
                #    print('current_labse_value', current_labse_value)
                    #print('source null', source[-1])
                if current_labse_value > labse_value:
                    labse_value = current_labse_value
                    best_node = '[' + ','.join(current_source) + ':' + ','.join(target) + ']'
            if (int(target[-1])+1) in target_nulls:
                #print('target null', target[-1])
                current_target = target + [str(int(target[-1])+1)]
                current_labse_value = get_labse_score(source, current_target, src_emb_dict, trg_emb_dict, source_dict, target_dict)
                #print(labse_value, current_labse_value)
                if current_labse_value > labse_value:
                    best_node = '[' + ','.join(source) + ':' + ','.join(current_target) + ']'
        except:
            pass
        splitpath[splitpath.index(i)] = best_node
    path = '\n'.join(splitpath)
    #print('path', path)
    return path

def get_labse_score(source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict):
    #print(source)
    s = ''
    for src in source:
        if len(src) > 0:
            s += source_dict[int(src)] + ' '
    s = s.strip()

    #print(target)
    t = ''
    for trg in target:
        if len(trg) > 0:
            t += target_dict[int(trg)] + ' '
    t = t.strip()

    try:
        trg_embeddings = trg_emb_dict[t]
    except KeyError as e:
        pass
        #print(e)
        # create embeddings for t

    try:
        src_embeddings = src_emb_dict[s]
    except KeyError as e:
        pass
        #print(e)
        # create embeddings for source_dict[int(source[0])]

    try:
        labse_score = trg_embeddings.dot(src_embeddings.transpose())
    except:
        labse_score = 0
    return labse_score


def create_combinations_from_concatenations(sentence_list):
    output = []
    for L in range(0, len(sentence_list) + 1):
        for subset in combinations(sentence_list, L):
            output.append(list(subset))
    output = output[1:]
    return output


def get_highest_scoring_pairs(source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff):
    source_combinations = create_combinations_from_concatenations(source)
    #print(source_combinations)
    target_combinations = create_combinations_from_concatenations(target)

    return_pairs = []
    max_score = 0
    max_combination = []
    for source_pair in source_combinations:
        #print(source_pair)
        source_embedding_exists = False
        # create a concatenated string from source_dict
        s = ''
        for src in source_pair:
            #print(src)
            s += source_dict[int(src)] + ' '
        s = s.strip()

        try:
            src = src_emb_dict[s]
            source_embedding_exists = True
        except KeyError as e:
            pass
            #print(e)
            # create embeddings for t

        if source_embedding_exists:
            for target_pair in target_combinations:
                t = ''
                for trg in target_pair:
                    t += target_dict[int(trg)] + ' '
                t = t.strip()

                try:
                    trg = trg_emb_dict[t]
                    score = trg.dot(src.transpose())
                    if score > max_score:
                        max_score = score
                        max_combination = [source_pair, target_pair]
                except KeyError as e:
                    pass
                    #print(e)

    if max_score >= score_cutoff:
        return_pairs.append(max_combination)

        if max_combination[0][0] > source[0]:
            if max_combination[1][0] > target[0]:
                before_source = source[:source.index(max_combination[0][0])]
                before_target = target[:target.index(max_combination[1][0])]
                further_pairs = get_highest_scoring_pairs(before_source, before_target, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff)
                if len(further_pairs) > 0:
                    return_pairs = further_pairs + return_pairs

        if max_combination[0][-1] < source[-1]:
            if max_combination[1][-1] < target[-1]:
                after_source = source[source.index(max_combination[0][-1]) + 1:]
                after_target = target[target.index(max_combination[1][-1]) + 1:]
                further_pairs = get_highest_scoring_pairs(after_source, after_target, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff)
                if len(further_pairs) > 0:
                    return_pairs = return_pairs + further_pairs
    return return_pairs


def reevaluate_path(path, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff):
    path_list = path.split('\n')[:-1]
    source_nulls = []
    target_nulls = []
    new_path = ''

    for i in path_list:
        current = i.strip('[]').split(':')
        source = current[0].split(',')
        target = current[1].split(',')
        if len(source) == 1 and len(target) == 1:
            new_path += i + '\n'
        elif len(source) > 1 and len(target) > 1:
            n_sentence_pairs = get_highest_scoring_pairs(source, target, src_emb_dict, trg_emb_dict, source_dict, target_dict, score_cutoff)
            if len(n_sentence_pairs) > 0:
                n_source_notnulls = []
                n_target_notnulls = []
                for pair in n_sentence_pairs:
                    new_path += '[' + ','.join(pair[0]) + ':' + ','.join(pair[1]) + ']\n'
                    n_source_notnulls += pair[0]
                    n_target_notnulls += pair[1]
                    target_nulls += list(set(target) - set(n_target_notnulls))
                    source_nulls += list(set(source) - set(n_source_notnulls))
        elif len(source) == 0:
            for trg in target:
                target_nulls.append(trg)
        elif len(target) == 0:
            for src in source:
                source_nulls.append(src)
        elif len(source) == 1 and len(target) > 1:
            target_combinations = []
            for L in range(0, len(target) + 1):
                for subset in combinations(target, L):
                    target_combinations.append(list(subset))
            target_combinations = target_combinations[1:]
            max_score = 0
            max_combination = []
            for combination in target_combinations:
                # create a concatenated string from target_dict
                t = ''
                for trg in combination:
                    t += target_dict[int(trg)] + ' '
                t = t.strip()

                # calculate labse_score for the pairs and return the highest scoring one, if there are extraneous numbers they are put in the list of nulls
                try:
                    trg_embeddings = trg_emb_dict[t]
                except KeyError as e:
                    pass
                    #print(e)
                    #create embeddings for t

                try:
                    src_embeddings = src_emb_dict[source_dict[int(source[0])]]
                except KeyError as e:
                    pass
                    #print(e)
                    #create embeddings for source_dict[int(source[0])]

                try:
                    labse_score = trg_embeddings.dot(src_embeddings.transpose())
                    #print('labse_score', labse_score)
                    if labse_score > max_score:
                        max_score = labse_score
                        max_combination = combination
                except Exception as e:
                    pass
                    #print(e)

            target_nulls += list(set(target) - set(max_combination))
            new_path += '[' + str(source[0]) + ':' + ','.join(max_combination).strip(',') + ']\n'

        elif len(source) > 1 and len(target) == 1:
            source_combinations = []
            for L in range(0, len(source) + 1):
                for subset in combinations(source, L):
                    source_combinations.append(list(subset))
            source_combinations = source_combinations[1:]
            max_score = 0
            max_combination = []
            for combination in source_combinations:
                # create a concatenated string from target_dict
                s = ''
                for src in combination:
                    s += source_dict[int(src)] + ' '
                s = s.strip()

                # calculate labse_score for the pairs and return the highest scoring one, if there are extraneous numbers they are put in the list of nulls
                try:
                    src_embeddings = src_emb_dict[s]
                except KeyError as e:
                    pass
                    #print(e)
                # create embeddings for t

                try:
                    trg_embeddings = trg_emb_dict[target_dict[int(target[0])]]
                except KeyError as e:
                    pass
                    #print(e)

                try:
                    labse_score = trg_embeddings.dot(src_embeddings.transpose())
                    if labse_score > max_score:
                        max_score = labse_score
                        max_combination = combination
                except Exception as e:
                    pass
                    #print(e)
            source_nulls += list(set(source) - set(max_combination))
            new_path += '[' + ','.join(max_combination).strip(',') + ':' + str(target[0]) + ']\n'

    return source_nulls, target_nulls, new_path


#fall sem skilar source language strengjum sem þarf að líta til þegar skor eru reiknuð. Fallið finnur öll concats fyrir tiltekinn punkt.
def concat_strings(start_node, x, max_concats, collection_length, collection_dict):
    if x <= collection_length+start_node:
        concat_array = []
        counter_array = []
        for i in range(max(start_node + 1,x + 1 - max_concats), x + 1):
            current_array = []
            ctr = 0
            for k in range(i, x+1):
                current_array.append(collection_dict[k])
                ctr += 1
            concat_array.append(' '.join(current_array))
            counter_array.append(ctr)
        yield concat_array, counter_array


def create_concats(start_node, position, max_concats, filelength, sentencedict):
    concats = []
    concats_count_dict = {}
    for string_array, count_array in concat_strings(start_node, position, max_concats, filelength, sentencedict):
        for strengir in range(0, len(string_array)):
            concats.append(string_array[strengir])
            concats_count_dict[string_array[strengir]] = count_array[strengir]
    return concats, concats_count_dict
