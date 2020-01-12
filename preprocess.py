#! /usr/bin/python3


import os
import argparse
import string

import checkpointing
import h5py
import numpy as np
from dataset import tokenize_data
from tqdm import tqdm
from collections import Counter

from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.msmarco import MSMARCO
from qa_utils.preprocessing.insrqa import InsuranceQA
from text_input import rich_tokenize


def tokenize(query, passage, token_to_id, char_to_id):
    q_tokens, q_chars, _, _, _ = \
        rich_tokenize(query, token_to_id, char_to_id, update=True)
    p_tokens, p_chars, _, _, mapping = \
        rich_tokenize(passage, token_to_id, char_to_id, update=True)
    return (query, (p_tokens, p_chars), (q_tokens, q_chars), passage, mapping)

def max_words_per_passage(sentences):
    c = 0
    for s in sentences:
        cc = len(s.translate(str.maketrans('', '', string.punctuation)).split(' '))
        if cc > c:
            c = cc
            #print(s)
        #c = cc if cc > c else c
    #print(c)
    return c


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

    token_to_id = {'': 0}
    char_to_id = {'': 0}

    p_word_count = max_words_per_passage(list(ds.docs.values()))

    # save h5 data
    var_int32 = h5py.special_dtype(vlen=np.dtype('int32'))
    dt = h5py.string_dtype(encoding='utf-8')

    train_file = os.path.join(args.SAVE, 'train.h5')
    print('writing {}...'.format(train_file))
    with h5py.File(train_file, 'w') as fp:
        # the length corresponds to pairs of positive and negative documents
        train_shape = (len(ds.trainset) * (1 + args.num_neg_examples), 7)
        token_chars_shape = (len(ds.trainset) * (1 + args.num_neg_examples), 2)
        #inputs_ds = fp.create_dataset('inputs', train_shape, dtype=var_int32)
        query_ds = fp.create_dataset('queries', train_shape, dtype=dt)
        p_token_chars_ds = fp.create_dataset('p_token_chars', train_shape, dtype=var_int32)
        q_token_chars_ds = fp.create_dataset('q_token_chars', train_shape, dtype=var_int32)
        passages_ds = fp.create_dataset('passages', train_shape, dtype=dt)
        mappings_ds = fp.create_dataset('mappings', train_shape, dtype=var_int32)
        labels_ds = fp.create_dataset('labels', train_shape, dtype='int32')
        i = 0
        for query, pos_doc, neg_docs in tqdm(ds.trainset):
            #inputs_ds[i] = tokenize(query, pos_doc, token_to_id, char_to_id)
            t_p = tokenize(query, pos_doc, token_to_id, char_to_id)
            query_ds[i] = t_p[0]
            p_token_chars_ds = t_p[1]
            q_token_chars_ds = t_p[2]
            passages_ds = t_p[3]
            mappings_ds = t_p[4]
            labels_ds[i] = 1
            i += 1
            for neg_doc in neg_docs:
                #inputs_ds[i] = tokenize(query, neg_doc, token_to_id, char_to_id)
                t_n = tokenize(query, neg_doc, token_to_id, char_to_id)
                query_ds[i] = t_n[0]
                p_token_chars_ds = t_n[1]
                q_token_chars_ds = t_n[2]
                passages_ds = t_n[3]
                mappings_ds = t_n[4]
                labels_ds[i] = 0
                i += 1
        # create and save vocabs
        id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
        id_to_char = {id_: char for char, id_ in char_to_id.items()}
        checkpointing.save_vocab(fp, 'vocab', id_to_token)
        checkpointing.save_vocab(fp, 'c_vocab', id_to_char)
        checkpointing.save_max_passage_length(p_word_count, fp)

    dev_file = os.path.join(args.SAVE, 'dev.h5')
    print('writing {}...'.format(dev_file))
    with h5py.File(dev_file, 'w') as fp:
        dev_shape = (len(ds.devset),)
        query_ds = fp.create_dataset('queries', dev_shape, dtype=dt)
        p_token_chars_ds = fp.create_dataset('p_token_chars', dev_shape, dtype=var_int32)
        q_token_chars_ds = fp.create_dataset('q_token_chars', dev_shape, dtype=var_int32)
        passages_ds = fp.create_dataset('passages', dev_shape, dtype=dt)
        mappings_ds = fp.create_dataset('mappings', dev_shape, dtype=var_int32)
        labels_ds = fp.create_dataset('labels', dev_shape, dtype='int32')
        for i, (q_id, query, doc, label) in enumerate(tqdm(ds.devset)):
            t_p = tokenize(query, pos_doc, token_to_id, char_to_id)
            query_ds[i] = t_p[0]
            p_token_chars_ds = t_p[1]
            q_token_chars_ds = t_p[2]
            passages_ds = t_p[3]
            mappings_ds = t_p[4]
            labels_ds[i] = 1
            i += 1
        # create and save vocabs
        id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
        id_to_char = {id_: char for char, id_ in char_to_id.items()}
        checkpointing.save_vocab(fp, 'vocab', id_to_token)
        checkpointing.save_vocab(fp, 'c_vocab', id_to_char)
        checkpointing.save_max_passage_length(p_word_count, fp)

    test_file = os.path.join(args.SAVE, 'test.h5')
    print('writing {}...'.format(test_file))
    with h5py.File(test_file, 'w') as fp:
        test_shape = (len(ds.testset),)
        query_ds = fp.create_dataset('queries', test_shape, dtype=dt)
        p_token_chars_ds = fp.create_dataset('p_token_chars', test_shape, dtype=var_int32)
        q_token_chars_ds = fp.create_dataset('q_token_chars', test_shape, dtype=var_int32)
        passages_ds = fp.create_dataset('passages', test_shape, dtype=dt)
        mappings_ds = fp.create_dataset('mappings', test_shape, dtype=var_int32)
        labels_ds = fp.create_dataset('labels', test_shape, dtype='int32')
        for i, (q_id, query, doc, label) in enumerate(tqdm(ds.testset)):
            t_p = tokenize(query, pos_doc, token_to_id, char_to_id)
            query_ds[i] = t_p[0]
            p_token_chars_ds = t_p[1]
            q_token_chars_ds = t_p[2]
            passages_ds = t_p[3]
            mappings_ds = t_p[4]
            labels_ds[i] = 1
            i += 1
        # create and save vocabs
        id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
        id_to_char = {id_: char for char, id_ in char_to_id.items()}
        checkpointing.save_vocab(fp, 'vocab', id_to_token)
        checkpointing.save_vocab(fp, 'c_vocab', id_to_char)
        checkpointing.save_max_passage_length(p_word_count, fp)


if __name__ == '__main__':
    main()
