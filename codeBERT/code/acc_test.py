from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
from collections import OrderedDict, Counter
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import T5Config, T5ForConditionalGeneration,RobertaTokenizer
import torch
from transformers import 
from eval.bleu import corpus_bleu
from eval.rouge import Rouge
from eval.meteor import Meteor

import run

MODEL_CLASSES = {'codeT5':(T5Config, T5ForConditionalGeneration,RobertaTokenizer)}


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    EM, precision, recall, f1 = 0, 0, 0, 0
    for gt in ground_truths:
        _EM, _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            EM, precision, recall, f1 = _EM, _prec, _rec, _f1
    return EM, precision, recall, f1


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            EM, precision, recall, f1 = 1, 1, 1, 1
    else:
        EM = (normalize_answer(prediction) == normalize_answer(ground_truth))
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return EM, precision, recall, f1


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


# with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
#         os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
#     for ref, gold in zip(p, eval_examples):
#         predictions.append(str(gold.idx) + '\t' + ref)
#         f.write(str(gold.idx) + '\t' + ref + '\n')  # ref is actually the hypothesis
#         f1.write(str(gold.idx) + '\t' + gold.target + '\n')
#         hypotheses[id] = [ref]
#         references[id] = [gold.target]
#         id += 1

# EM, bleu4, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
#                                                                     references)
#

import  run

from  run import eval_accuracies
if __name__ == "__main__":
    answer_file = ""
    question_file = ""
    references_file = ""

    answers = open(answer_file,"r").readlines()
    questions = open(question_file,"r").readlines()
    references = open(references_file,"r").readlines()
    
    data = defaultdict(list)

    for a,q,r in zip(answers,questions,references_file):
        if "for what purpose" in q:
            data["for what purpose"].append([a,r])
        elif "what" in q:
            data["what"].append([a,r])
        elif "how" in q:
            data["what"].append([a,r])
        elif "when" in q:
            data["when"].append([a,r])
        elif "where" in q:
            data["where"].append([a,r])
        else:
            data["other"].append([a,r])


    predictions = []
    id = 0
    hypotheses, references = dict(), dict()
    for ref in cc_file2:
        hypotheses[id] = [ref]
        id += 1
    id  = 0
    for gold in cc_file:
        references[id] = [gold]
        id += 1
    print(len(references.keys()))
    print(len(hypotheses.keys()))
    EM, bleu4, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,references)
    print(bleu4)