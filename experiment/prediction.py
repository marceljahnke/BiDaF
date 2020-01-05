#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import random
import yaml
import argparse
import os.path
import re
import numpy as np
import torch
import h5py
from bidaf import BidafModel
from experiment.dataset import load_data, tokenize_data, EpochGen
from experiment.dataset import SymbolEmbSourceNorm
from experiment.dataset import SymbolEmbSourceText
from experiment.dataset import symbol_injection

from import_scripts import fiqa, ms_marco

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')


def try_to_resume(exp_folder):
    if os.path.isfile(exp_folder + '/checkpoint'):
        checkpoint = h5py.File(exp_folder + '/checkpoint')
    else:
        checkpoint = None
    return checkpoint


def reload_state(checkpoint, config, args):
    """
    Reload state before predicting.
    """
    print('Loading Model...')
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    #with open(args.data) as f_o:
    #    data, _ = load_data(json.load(f_o), span_only=True, answered_only=True)

    # --------- load TSVs as pandas data frames
    # --------- FiQA
    path_to_passages = './data/fiqa/FiQA_train_doc_final.tsv'
    path_to_queries = './data/fiqa/FiQA_train_question_final.tsv'
    path_to_relevance = './data/fiqa/FiQA_train_question_doc_final.tsv'
    data = fiqa.load_data(path_to_passages, path_to_queries, path_to_relevance)
    # --------- MS MARCO
    # path_to_passages = './data/ms_marco/collection.tsv'
    # path_to_queries = './data/ms_marco/queries.train.tsv'
    # path_to_relevance = './data/ms_marco/qrels.train.tsv'
    # data = ms_marco.load_data(path_to_passages, path_to_queries, path_to_relevance)
    # --------------- Split data into training and test data
    data = data.iloc[int(len(data.index) * 0.8):]  # test
    print('Generated positive and negative examples: ', len(data.index))
    # ---------- done loading data

    print('Tokenizing data...')
    data = tokenize_data(data, token_to_id, char_to_id)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    data = get_loader(data, args)

    if len_tok_voc != len(token_to_id):
        need = set(tok for id_, tok in id_to_token.items()
                   if id_ >= len_tok_voc)

        if args.word_rep:
            with open(args.word_rep) as f_o:
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
        model.cuda()
    model.eval()

    return model, id_to_token, id_to_char, data


def get_loader(data, args):
    data = EpochGen(
        data,
        batch_size=args.batch_size,
        shuffle=False)
    return data


def predict(model, data):
    """
    Train for one epoch.
    """
    #for batch_id, (qids, passages, queries, _, mappings) in enumerate(data):
    for batch_id, (qids, passages, queries, relevances, mappings) in enumerate(data):
        predicted_relevance = model(passages[:2], passages[2], queries[:2], queries[2])
        #predictions = model.get_best_span(start_log_probs, end_log_probs)
        predictions = predicted_relevance.cpu()
        passages = passages[0].cpu().data
        for qid, query, mapping, tokens, pred, rel in zip(qids, queries, mappings, passages, predictions, relevances):
            yield (qid, query, tokens, pred, rel)
    return


def main():
    """
    Main prediction program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Prediction data")
    argparser.add_argument("dest", help="Write predictions in")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                                "word representations.")
    argparser.add_argument("--batch_size",
                           type=int, default=64,
                           help="Batch size to use")
    argparser.add_argument("--cuda",
                           type=bool, default=False,
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                                "when generating random word representations.")

    args = argparser.parse_args()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint = try_to_resume(args.exp_folder)

    if checkpoint:
        model, id_to_token, id_to_char, data = reload_state(
            checkpoint, config, args)
    else:
        print('Need a valid checkpoint to predict.')
        return

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor
    qid2candidate = {}
    #for qid, toks, start, end in predict(model, data):
    for qid, query, toks, pred, rel in predict(model, data):
        '''
        toks = regex_multi_space.sub(' ', regex_drop_char.sub(' ', ' '.join(
            id_to_token[int(tok)] for tok in toks).lower())).strip()
        # print(repr(qid), repr(toks), start, end, file=f_o)
        output = '{\"query_id\": ' + qid + ',\"answers\":[ \"' + toks + '\"]}'
        '''
        #print(f"Types of:\nqid: {type(qid)}\nquery: {type(query)}\ntokens; {type(tokens)}\npred: {type(pred)}\nrel: {type(rel)}")
        toks = regex_multi_space.sub(' ', regex_drop_char.sub(' ', ' '.join(id_to_token[int(tok)] for tok in toks).lower())).strip()
        #query = regex_multi_space.sub(' ', regex_drop_char.sub(' ', ' '.join(id_to_token[int(q)] for q in query).lower())).strip()
        #output = "{\"query_id\": " + str(qid) + ", \"passage\": [\"" + toks + "\"], \"predicted relevance\":  " + str(pred.item()) + ", \"actual relevance\": " + str(rel.item()) + "}"
        output = "{\"query_id\": " + str(qid) + ", \"p_rel\": " + str(pred.item()) + ", \"a_rel\": " + str(rel.item()) + "}"
        print(output)
        if qid not in qid2candidate:
            qid2candidate[qid] = []
        qid2candidate[qid].append(json.dumps(json.loads(output)))

    result = {'result': qid2candidate}
    with open(args.dest + 'predictions.txt', 'w') as file:
        file.write(json.dumps(result))

    #with open(args.dest, 'w') as f_o:
    #    for qid in qid2candidate:
    #        pick = random.randint(0, len(qid2candidate[qid]) - 1)
    #        f_o.write(qid2candidate[qid][pick])
    #        f_o.write('\n')
    print('Prediction done')
    return


if __name__ == '__main__':
    # run with: python experiment/prediction.py --word_rep ./data/glove.840B.300d.txt ./results/ ./data/ ./results/
    main()
