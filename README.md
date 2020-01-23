# BiDaF Baseline

This project is part of the AI laboratory of the Leibniz university Hanover (WiSe 19/20).

## Task
Given a question and a set of passages, rank them by relevance to the
question using the **BiDaF** model, for the following data sets:

- MS MARCO Passage Retrieval
- [FiQA](https://sites.google.com/view/fiqa)
- [InsuranceQA](https://github.com/shuzi/insuranceQA)
- [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)

The resulting data wil be used as a baseline in further studies. In addition, it is desirable that the model can easily be extended by further data sets.

## Requirements
The following packages are required to run the scripts:

- tqdm
- h5py
- numpy
- pytorch
- cuda
- yaml
- nltk

You also have to download the Global Vectors for Word Representation since they're used in the Char Embedding by the Bidaf model:

- [glove.840B.300d.txt](https://nlp.stanford.edu/data/glove.840B.300d.zip)

## Preprocessing
You need to preprocess the dataset you want to use before you can start the training or evaluation. Simply download the datset and start the preprocess script to preprocess the data. We're using a Framework provided by the L3S which requires specific arguments for the different dataset. Examples are given below:

MS MARCO Passage Retrieval:

  `python3 path/to/preprocess.py msmarco path/to/msmarco_dataset`

FiQA:

  `python3 path/to/preprocess.py fiqa path/to/fiqa_dataset path/to/BiDaF/qa_utils/splits/fiqa_split.pkl`

InsuranceQA:

  `python3 path/to/preprocess.py insrqa path/to/insrqa_dataset --examples_per_query [100, 500, 1000, 1500]`

ANTIQUE:

  `python3 path/to/preprocess.py antique path/to/antique_dataset path/to/BiDaF/qa_utils/splits/antique_split.pkl`

## Training
For the training you need use the preprocessed files. You can start the training by running the training script via:

```
python3 path/to/training.py path/to/Bidaf/experiment path/to/BiDaF/data path/to/Bidaf/results --word_rep path/to/glove.840B.300d.txt --seed 12345
```

You can change the configuration of the Bidaf model (e.g. number of epochs, batch size) by changing the `config.yaml` inside the experiment folder.

## Prediction
The evaluation uses the preprocessed files as well. You can start the evaluating of the trained model by running the prediction script via:

```
python3 path/to/prediction.py path/to/Bidaf/experiment path/to/BiDaF/data path/to/Bidaf/results --word_rep path/to/glove.840B.300d.txt --mrr_k k --seed 12345
```

The result will be written as a csv file in the `results` folder, containing the information about MRR@K and MAP.

## Results

| Dataset     | Dev MAP | Dev MRR@10 | Test MAP | Test MRR@10 |
|:------------|--------:| ----------:|---------:|------------:|
| MS MARCO    | tba     | tba        | tba      | tba         |
| FiQA        | 0.5774  | 0.7666     | 0.5761   | 0.7524      |
| InsuranceQA | tba     | tba        | tba      | tba         |
| ANTIQUE     | 0.4641  | 0.7318     | 0.5793   | 0.8251      |

## Objectives

- Implement the BiDAF model
- Modify the output layer to produce a score of a passage w.r.t. a query
- Implement ranking metrics (MRR, MAP)
- Good documentation of the code
- (possibility to extend the project with other datasets)
