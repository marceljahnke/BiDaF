#! /usr/bin/python3


import os
import argparse

import experiment.checkpointing as checkpointing
import h5py
import json
import numpy as np
from tqdm import tqdm

from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.msmarco import MSMARCO
from qa_utils.preprocessing.insrqa import InsuranceQA
from qa_utils.preprocessing.antique import Antique
from text_input import rich_tokenize

p_tokenized = {}
q_tokenized = {}


def tokenize_train(trainset, token_to_id, char_to_id):
    tokenized = []
    c = 0
    for query, pos_doc, neg_docs in tqdm(trainset):
        if query in q_tokenized:
            q_tokens, q_chars = q_tokenized[query]
        else:
            q_tokens, q_chars, _, _, _ = rich_tokenize(query, token_to_id, char_to_id, update=True)
            q_tokenized[query] = (q_tokens, q_chars)

        def tokenize_passage(passage, counter):
            if passage in p_tokenized:
                tokens, chars = p_tokenized[passage]
            else:
                tokens, chars, _, _, _ = rich_tokenize(passage, token_to_id, char_to_id, update=True)
                p_tokenized[passage] = (tokens, chars)
                if len(tokens) > counter:
                    counter = len(tokens)
            return tokens, chars, counter

        p_tokens, p_chars, c = tokenize_passage(pos_doc, c)
        tokenized.append(((p_tokens, p_chars), (q_tokens, q_chars), 1))

        for neg_doc in neg_docs:
            p_tokens, p_chars, c = tokenize_passage(neg_doc, c)
            tokenized.append(((p_tokens, p_chars), (q_tokens, q_chars), 0))

    return tokenized, c


def tokenize(data, token_to_id, char_to_id):
    tokenized = []
    c = 0
    for qid, query, passage, label in tqdm(data):
        if query in q_tokenized:
            q_tokens, q_chars = q_tokenized[query]
        else:
            q_tokens, q_chars, _, _, _ = rich_tokenize(query, token_to_id, char_to_id, update=True)
            q_tokenized[query] = (q_tokens, q_chars)

        if passage in p_tokenized:
            p_tokens, p_chars = p_tokenized[passage]
        else:
            p_tokens, p_chars, _, _, _ = rich_tokenize(passage, token_to_id, char_to_id, update=True)
            p_tokenized[passage] = (p_tokens, p_chars)
            if len(p_tokens) > c:
                c = len(p_tokens)

        tokenized.append((qid, (p_tokens, p_chars), (q_tokens, q_chars), label))
    return tokenized, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('SAVE', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=1, help='Number of negative examples to sample')

    subparsers = ap.add_subparsers(help='Choose a dataset', dest='dataset')
    subparsers.required = True
    FiQA.add_subparser(subparsers, 'fiqa')
    MSMARCO.add_subparser(subparsers, 'msmarco')
    InsuranceQA.add_subparser(subparsers, 'insrqa')
    args = ap.parse_args()

    os.makedirs(args.SAVE, exist_ok=True)
    if args.dataset == 'fiqa':
        ds = FiQA(args)
    elif args.dataset == 'insrqa':
        ds = InsuranceQA(args)
    elif args.dataset == 'msmarco':
        ds = MSMARCO(args)
    elif args.dataset == 'antique':
        ds = Antique(args)
    else:
        raise Exception('You must specify a valid dataset {fiqa, insrqa, msmarco, antique}!')

    token_to_id = {'': 0}
    char_to_id = {'': 0}

    tokenized_train, word_count_train = tokenize_train(ds.trainset, token_to_id, char_to_id)
    tokenized_dev, word_count_dev = tokenize(ds.devset, token_to_id, char_to_id)
    tokenized_test, word_count_test = tokenize(ds.testset, token_to_id, char_to_id)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}
    p_word_count = max(word_count_train, word_count_dev, word_count_test)

    # save h5 data
    dt = h5py.string_dtype(encoding='utf-8')
    dt_int_list = h5py.vlen_dtype(np.dtype('int32'))

    train_file = os.path.join(args.SAVE, 'train.h5')
    print('writing {}...'.format(train_file))
    with h5py.File(train_file, 'w') as fp:
        # the length corresponds to pairs of positive and negative documents
        train_shape = (len(ds.trainset) * (1 + args.num_neg_examples),)
        p_tokens_ds = fp.create_dataset('p_tokens', train_shape, dtype=dt_int_list)
        p_chars_ds = fp.create_dataset('p_chars', train_shape, dtype=dt)
        q_tokens_ds = fp.create_dataset('q_tokens', train_shape, dtype=dt_int_list)
        q_chars_ds = fp.create_dataset('q_chars', train_shape, dtype=dt)
        labels_ds = fp.create_dataset('labels', train_shape, dtype='int32')
        i = 0
        for ((p_tokens, p_chars), (q_tokens, q_chars), label) in tqdm(tokenized_train):
            p_tokens_ds[i] = p_tokens
            p_chars_ds[i] = json.dumps(p_chars)
            q_tokens_ds[i] = q_tokens
            q_chars_ds[i] = json.dumps(q_chars)
            labels_ds[i] = label
            i += 1
        checkpointing.save_max_passage_length(p_word_count, fp)
        checkpointing.save_vocab(fp, 'vocab', id_to_token)
        checkpointing.save_vocab(fp, 'c_vocab', id_to_char)

    def generate_ds(file, dataset):
        print('writing {}...'.format(file))
        with h5py.File(file, 'w') as fp:
            dev_shape = (len(dataset),)
            qid_ds = fp.create_dataset('qids', dev_shape, dtype='int32')
            p_tokens_ds = fp.create_dataset('p_tokens', dev_shape, dtype=dt_int_list)
            p_chars_ds = fp.create_dataset('p_chars', dev_shape, dtype=dt)
            q_tokens_ds = fp.create_dataset('q_tokens', dev_shape, dtype=dt_int_list)
            q_chars_ds = fp.create_dataset('q_chars', dev_shape, dtype=dt)
            labels_ds = fp.create_dataset('labels', dev_shape, dtype='int32')
            i = 0
            for (q_id, (p_tokens, p_chars), (q_tokens, q_chars), label) in tqdm(dataset):
                qid_ds[i] = q_id
                p_tokens_ds[i] = p_tokens
                p_chars_ds[i] = json.dumps(p_chars)
                q_tokens_ds[i] = q_tokens
                q_chars_ds[i] = json.dumps(q_chars)
                labels_ds[i] = label
                i += 1
            checkpointing.save_max_passage_length(p_word_count, fp)
            checkpointing.save_vocab(fp, 'vocab', id_to_token)
            checkpointing.save_vocab(fp, 'c_vocab', id_to_char)

    dev_file = os.path.join(args.SAVE, 'dev.h5')
    generate_ds(dev_file, tokenized_dev)
    test_file = os.path.join(args.SAVE, 'test.h5')
    generate_ds(test_file, tokenized_test)


if __name__ == '__main__':
    main()
