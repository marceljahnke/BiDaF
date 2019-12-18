import numpy as np
import pandas as pd
from pandas import DataFrame


def load_data(path_to_passages: str, path_to_queries: str, path_to_relevance: str):
    """
    Load the data and use it to create one example for a positive and one for
    a negative query + passage.
    Code could be shorter. First version contains more code for easier understanding.
    """
    passages = pd.read_csv(path_to_passages, sep='\t', names=['pid', 'passage'],encoding='utf-8')
    queries = pd.read_csv(path_to_queries, sep='\t', names=['qid', 'query'], encoding='utf-8')
    relevance = pd.read_csv(path_to_relevance, sep='\t', names=['qid', 'trec0', 'pid', 'trec1'], encoding='utf-8')

    np.random.seed(1234)

    # passages = passages.sort_values(by=['pid'], kind='heapsort') # überhaupt nötig oder schon vorsortiert?
    # queries = queries.sort_values(by=['qid'], kind='heapsort') # beschleunigt zugriff nicht, da Queries geteilt ?
    relevance = relevance.sort_values(by=['qid', 'pid'])  # sort via mergesort (standard for multiple cols)

    relevance_rows = len(relevance.index)

    # rows = number of rows in relevance * 2
    # columns = qid, query, pid, passage, relevance score
    data = []

    i = 0
    while i < relevance_rows:
        qid = relevance.iloc[i]['qid']
        query = queries.loc[queries['qid'] == qid].iloc[0]['query']
        j = i
        rel_passages = set()
        # check all relevant passages for one query before we generate negative example
        while True:
            pid = relevance.iloc[j]['pid']
            passage = passages.iloc[pid]['passage']
            rel_passages.add((pid, passage))
            j += 1
            if j >= relevance_rows or relevance.iloc[j]['qid'] != qid:
                break

        # update data positive examples
        for pid, passage in rel_passages:
            # data row: [qid, query, pid, passage, relevance score]
            # data.append((qid, query, pid, passage, 1))
            data.append((query, passage, 1))
            i += 1

        # index i is now at the position of the qid in the relevance table

        # generate negative examples
        not_rel_passages = np.setdiff1d(passages['pid'].to_numpy(), np.asarray([p[0] for p in rel_passages]))
        not_rel_passages = np.random.choice(not_rel_passages, len(rel_passages), replace=False)
        for pid in not_rel_passages:
            pid = pid
            passage = passages.iloc[pid]['passage']
            # data.append((qid, query, pid, passage, 0))
            data.append((query, passage, 0))

    # generate dataframe
    data = pd.DataFrame(data, columns=['query', 'passage',
                                       'relevance'])  # cheaper to append to list and create data frame in one go

    print(data)
    data.to_csv("./ms_marco_data_frame.tsv", sep="\t", encoding='utf-8', index=False)

    # shuffle data (in-place and reset indices, while preventing extra column with old indices)
    # data = data.sample(frac=1).reset_index(drop=True)
    return data
