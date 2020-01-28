#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import random
from collections import defaultdict

import yaml
import argparse
import os.path
import re
import numpy as np
import torch
import h5py
import math
import sys
import pandas as pd
from bidaf import BidafModel
from tqdm import tqdm
from experiment.dataset import load_data_from_h5, tokenize_data, EpochGen
from experiment.dataset import SymbolEmbSourceNorm
from experiment.dataset import SymbolEmbSourceText
from experiment.dataset import symbol_injection

from qa_utils.evaluation import get_ranking_metrics
from qa_utils.misc import Logger
from sklearn.metrics import accuracy_score

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')


def try_to_resume(exp_folder):
    if os.path.isfile(exp_folder + '/checkpoint'):
        checkpoint = h5py.File(exp_folder + '/checkpoint')
    else:
        checkpoint = None
    return checkpoint


def reload_state(checkpoint, config, args, file):
    """
    Reload state before predicting.
    """

    print('Loading Model...')
    model, id_to_token, id_to_char, _ = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)

    print('Loading data...')
    data, _, _, _ = load_data_from_h5(file, use_dummy_qids=False)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    data = get_loader(data, args)
    print("data got loaded")

    if len_tok_voc != len(token_to_id):
        need = set(tok for id_, tok in id_to_token.items()
                   if id_ >= len_tok_voc)

        if args.word_rep:
            with open(args.word_rep, encoding='utf-8') as f_o:
                pre_trained = SymbolEmbSourceText(
                    f_o, need)
        else:
            pre_trained = SymbolEmbSourceText([], need)

        cur = model.embedder.embeddings[0].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        if args.word_rep:
            print('Augmenting with pre-trained embeddings...')
        else:
            print('Augmenting with random embeddings...')

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, len_tok_voc,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if len_char_voc != len(char_to_id):
        print('Augmenting with random char embeddings...')
        pre_trained = SymbolEmbSourceText([], None)
        cur = model.embedder.embeddings[1].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[1].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_char, len_char_voc,
                model.embedder.embeddings[1].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if torch.cuda.is_available() and args.cuda:
        model.to(torch.device('cuda:0'))
        model = torch.nn.DataParallel(model)
    model.eval()

    return model, id_to_token, id_to_char, data


def get_loader(data, args):
    data = EpochGen(
        data,
        batch_size=args.batch_size,
        shuffle=False)
    return data


def evaluate(model, dataloader, k, device):
    result = defaultdict(lambda: ([], []))
    for batch in tqdm(dataloader):
        qids, passages, queries, labels = batch
        predictions = model(passages[:2], passages[2], queries[:2], queries[2]).cpu().detach()
        for q_id, prediction, label in zip(qids, predictions, labels):
            result[q_id][0].append(prediction)
            result[q_id][1].append(label)

    all_scores, all_labels = [], []
    for q_id, (score, label) in result.items():
        all_scores.append(score)
        all_labels.append(label)
    return get_ranking_metrics(all_scores, all_labels, k)

def main():
    """
    Main prediction program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Prediction data")
    argparser.add_argument("dest", help="Write predictions in")
    argparser.add_argument('--mrr_k', type=int, default=10, help='Compute MRR@k')
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                                "word representations.")
    argparser.add_argument("--batch_size",
                           type=int, default=64,
                           help="Batch size to use")
    argparser.add_argument("--cuda",
                           type=bool, default=torch.cuda.is_available(),
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                                "when generating random word representations.")
    argparser.add_argument("--seed", default=12345, help="Seed for Pytorch")
    argparser.add_argument("--multi_gpu", default=True, help="Use multiple GPUs for evaluation")

    args = argparser.parse_args()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    torch.manual_seed(args.seed)
    checkpoint = try_to_resume(args.dest)

    if not checkpoint:
        print('Need a valid checkpoint to predict.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = True
    dev = True

    if test:
        model, id_to_token, id_to_char, test_dl = reload_state(checkpoint, config, args, file='/home/jahnke/BiDaF/data/preprocessed/test.h5')
        if torch.cuda.is_available() and args.cuda:
            test_dl.tensor_type = torch.cuda.LongTensor
            with torch.no_grad():
                test_metrics = evaluate(model, test_dl, args.mrr_k, device)
        print('Done test set')

    if dev:
        model, id_to_token, id_to_char, dev_dl = reload_state(checkpoint, config, args,
                                                               file='/home/jahnke/BiDaF/data/preprocessed/dev.h5')
        if torch.cuda.is_available() and args.cuda:
            dev_dl.tensor_type = torch.cuda.LongTensor
            with torch.no_grad():
                dev_metrics = evaluate(model, dev_dl, args.mrr_k, device)
        print('Done dev set')

    eval_file = os.path.join(args.dest, 'eval.csv')
    logger = Logger(eval_file, ['dev_map', 'dev_mrr', 'test_map', 'test_mrr'])
    metrics = list(dev_metrics) + list(test_metrics)
    logger.log(metrics)

    print('Evaluation done')
    return


if __name__ == '__main__':
    # run with: python experiment/prediction.py --word_rep ./data/glove.840B.300d.txt ./experiment/ ./data/ ./results/
    main()
