#! /usr/bin/python3


import os
import argparse

import experiment.checkpointing as checkpointing
import h5py
from tqdm import tqdm

from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.msmarco import MSMARCO
from qa_utils.preprocessing.insrqa import InsuranceQA
from qa_utils.preprocessing.antique import Antique
from text_input import rich_tokenize


def max_words_per_passage(queries, docs):
    token_to_id = {'': 0}
    char_to_id = {'': 0}
    c = 0
    for query in queries:
        q_tokens, q_chars, _, _, _ = rich_tokenize(query, token_to_id, char_to_id, update=True)
    for doc in docs:
        p_tokens, p_chars, _, _, mapping = rich_tokenize(doc, token_to_id, char_to_id, update=True)
        if len(p_tokens) > c:
            c = len(p_tokens)
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
    Antique.add_subparser(subparsers, 'antique')
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

    p_word_count = max_words_per_passage(list(ds.queries.values()), list(ds.docs.values()))

    # save h5 data
    dt = h5py.string_dtype(encoding='utf-8')

    train_file = os.path.join(args.SAVE, 'train.h5')
    print('writing {}...'.format(train_file))
    with h5py.File(train_file, 'w') as fp:
        # the length corresponds to pairs of positive and negative documents
        train_shape = (len(ds.trainset) * (1 + args.num_neg_examples),)
        query_ds = fp.create_dataset('queries', train_shape, dtype=dt)
        passages_ds = fp.create_dataset('passages', train_shape, dtype=dt)
        labels_ds = fp.create_dataset('labels', train_shape, dtype='int32')
        i = 0
        for query, pos_doc, neg_docs in tqdm(ds.trainset):
            query_ds[i] = query
            passages_ds[i] = pos_doc
            labels_ds[i] = 1
            i += 1
            for neg_doc in neg_docs:
                query_ds[i] = query
                passages_ds[i] = neg_doc
                labels_ds[i] = 0
                i += 1
        checkpointing.save_max_passage_length(p_word_count, fp)

    dev_file = os.path.join(args.SAVE, 'dev.h5')
    print('writing {}...'.format(dev_file))
    with h5py.File(dev_file, 'w') as fp:
        dev_shape = (len(ds.devset),)
        print(dev_shape)
        qid_ds = fp.create_dataset('qids', dev_shape, dtype='int32')
        query_ds = fp.create_dataset('queries', dev_shape, dtype=dt)
        passages_ds = fp.create_dataset('passages', dev_shape, dtype=dt)
        labels_ds = fp.create_dataset('labels', dev_shape, dtype='int32')
        i = 0
        for _, (q_id, query, doc, label) in enumerate(tqdm(ds.devset)):
            qid_ds[i] = q_id
            query_ds[i] = query
            passages_ds[i] = doc
            labels_ds[i] = label
            i += 1
        checkpointing.save_max_passage_length(p_word_count, fp)

    test_file = os.path.join(args.SAVE, 'test.h5')
    print('writing {}...'.format(test_file))
    with h5py.File(test_file, 'w') as fp:
        test_shape = (len(ds.testset),)
        qid_ds = fp.create_dataset('qids', test_shape, dtype='int32')
        query_ds = fp.create_dataset('queries', test_shape, dtype=dt)
        passages_ds = fp.create_dataset('passages', test_shape, dtype=dt)
        labels_ds = fp.create_dataset('labels', test_shape, dtype='int32')
        i = 0
        for _, (q_id, query, doc, label) in enumerate(tqdm(ds.testset)):
            qid_ds[i] = q_id
            query_ds[i] = query
            passages_ds[i] = doc
            labels_ds[i] = label
            i += 1
        checkpointing.save_max_passage_length(p_word_count, fp)


if __name__ == '__main__':
    main()
