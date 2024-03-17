import re
import os
import ast
import pickle
import string
import collections

import nltk
import transformers as trf
import numpy as np
import pandas as pd

from utils import load_dataset, compute_density, KMPSearch, encode, decode

import scipy
from sklearn import metrics
from sklearn import linear_model
from matplotlib import pyplot as plt
import seaborn as sns



def match_instruct_context(pos, prompt, instruct, tokens, tokenizer):
    masks = [0.] * len(tokens)
    for sid, span in enumerate(instruct.split("|||"), 1):
        if len(span) == 0:
            continue
        assert span in prompt, str(pos) + span + "|||" + prompt
        if prompt[max(0,  prompt.index(span) - 1)] == "\n":
            span = "\n" + span
        span = tokenizer.tokenize(span)
        while len(span) > 0 and span[0] in {u"\u2581", "<0x0A>", u'\u2581"'}:
            span = span[1:]
        while len(span) > 0 and span[-1] in {u"\u2581", "<0x0A>", u'\u2581"'}:
            span = span[:-1]
        if len(span) > 0:
            idx = KMPSearch(span, tokens)
            assert idx >= 0, str(pos) + str(span) + str(tokens)
            masks[idx:idx + len(span)] = [1.0] * len(span)
    return np.array(masks)


def evaluate(idx, prompt, annotate, rslt, tokenizer, name, eps=7):
    if len(rslt["Output"]) == 0:
        return (0.0, 0.0)
    inst_span, cntx_span = annotate.split("&&&") if "&&&" in annotate else (annotate, "")
    inst_mask = match_instruct_context(idx, prompt, inst_span, rslt["Input"], tokenizer)
    compare = compute_density(rslt["Explain"], eps) * inst_mask
    inst_score = (compare * inst_mask).sum() / np.log2(2 + inst_mask.sum())
    return inst_score



def load_result_annotation(root, annotate_file, model_name, IDs, Prompt, Reference, Instruct):
    with open(annotate_file, "r", encoding="utf8") as f:
        for row in f:
            file, model, prompt, ref, instruction, response, rating = row[:-1].split("\t")
            if model_name == model:
                if not os.path.exists(root + "/" + file):
                    continue
                rslt = pickle.load(open(root + "/" + file, "rb"))
                idx = file.rsplit(".", 1)[0]
                if idx in IDs:
                    idx = IDs.index(idx)
                    yield file, decode(Prompt[idx]), rslt, decode(Instruct[idx]), float(rating)
    


if __name__ == "__main__":
    tokenizer_path = r"lmsys/vicuna-7b-v1.5" # The tokenizer has no changed.
    rating_file = r"../datasets/AnnotatedDataset.tsv"
    dataset_path = r"../datasets/FinalDataset.tsv"
    root = r"../results/attribution_interpret\ifa_"
    ID, Prompt, Reference, Instruct = load_dataset(dataset_path)
    tokenizer = trf.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    model_name = "vicuna_7B"
    model_root = root + model_name
    inst_scores, real_scores, groups = [], [], []
    input_lens, output_lens = [], []
    for file, prompt, rslt, instruction, rating in load_result_annotation(model_root, rating_file, model_name,
                                                                          ID, Prompt, Reference, Instruct):
        if float(rating) <= 0:
            continue
        if len(rslt["Output"]) < 5:
            continue
        if rating >= 2:
            rating = 2
        predict = evaluate(file, prompt, instruction, rslt, tokenizer, model_name)
        inst_scores.append(predict)
        real_scores.append(rating)
        groups.append(file.split("_")[0])
    unique_groups = set(groups)
    groups = np.array(groups)
    inst_scores = np.array(inst_scores)
    real_scores = np.array(real_scores)
        
            
        measurement = lambda x, y: scipy.stats.ttest_ind(x, y, equal_var=False, alternative="less")
        tags = sorted(set(real_scores))
        for group in unique_groups:
            choose = groups == group
            group_rate = real_scores[choose]
            group_score = inst_scores[choose]
            print()
            print('GIA %s:' % group, measurement(group_score[group_rate==1], group_score[group_rate==2]))
            print("Followed Percentage:", [(group_rate == l).sum() / len(group_rate) for l in tags])
            print("Average:", [group_score[group_rate == l].mean() for l in tags])
            print("StandardDeviation:", [group_score[group_rate == l].std() for l in tags])

