import numpy as np
import pandas as pd


def load_data(path_to_passages: str, path_to_queries: str, path_to_relevance: str):
    """
    Load the data and use it to create one example for a positive and one for
    a negative query + passage.
    Code could be shorter. First version contains more code for easier understanding.

    Parameters:
        :param: path_to_passages: path to tsv file with qid, query and timestamp
        :param: path_to_queries: path to tsv file with docid (= pid), passage and timestamp
        :param: path_to_relevance: path to tsv file with qid and docid of relevant passage

    Returns:
        :return: table with the form [n x 3], number of rows are equal to rows of relevance * 2, columns are query, passage and relevance
    """

    # import and filter data
    passages = pd.read_csv(path_to_passages, sep='\t', header=0, names=['index', 'pid', 'passage', 'timestamp'], encoding='utf-8')
    passages = passages.drop(columns=['index', 'timestamp'])
    nan_pids = (passages.loc[passages['passage'].isnull()])['pid'] # pandas series
    passages = passages.dropna()

    queries = pd.read_csv(path_to_queries, sep='\t', header=0, names=['index', 'qid', 'query', 'timestamp'], encoding='utf-8')
    queries = queries.drop(columns=['index', 'timestamp'])
    # nan_qids = (queries.loc[queries['query'].isnull()])['qid']  # empty series
    queries = queries.dropna()

    relevance = pd.read_csv(path_to_relevance, sep='\t', header=0, names=['index', 'qid', 'pid'], encoding='utf-8')
    relevance = relevance.drop(columns=['index'])
    # relevance = relevance.dropna()
    relevance = relevance[~relevance['pid'].isin(nan_pids)] # filter rows with relevant nan passages

    np.random.seed(1234)

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
            # kein direkter Zugriff möglich, da FiQA Passagen Lücken enthält
            passage = passages.loc[passages['pid'] == pid].iloc[0]['passage']
            rel_passages.add((pid, passage))
            j += 1
            if j >= relevance_rows or relevance.iloc[j]['qid'] != qid:
                break

        # update data positive examples
        for pid, passage in rel_passages:
            data.append((qid, query, passage, 1))
            i += 1

        # index i is now at the position of the qid in the relevance table

        # generate negative examples
        not_rel_passages = np.setdiff1d(passages['pid'].to_numpy(), np.asarray([p[0] for p in rel_passages]))
        not_rel_passages = np.random.choice(not_rel_passages, len(rel_passages), replace=False)
        for pid in not_rel_passages:
            pid = pid
            passage = passages.loc[passages['pid'] == pid].iloc[0]['passage']
            data.append((qid, query, passage, 0))

    # generate dataframe
    data = pd.DataFrame(data, columns=['qid', 'query', 'passage',
                                       'relevance'])  # cheaper to append to list and create dataframe in one go

    if data.isnull().values.any():
        print("contains nan values")
    # shuffle data (in-place and reset indices, while preventing extra column with old indices)
    # data.sample(frac=1).reset_index(drop=True)
    return data
