"""
INTRO: Word2Vec algorithm

Distributed word vectors.

Word2vec is a neural network implementation (written in C) that learns distributed representations for words and
learns quickly relative to other models. Word2Vec also does NOT need labels in order to create meaningful representations.
This is useful, since most data in the real world is unlabeled. If the network is given enough training data (tens of
billions of words), it produces word vectors with intriguing characteristics. Words with similar meanings appear in clusters,
and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math.
The famous example is that, with highly trained word vectors, "king - man + woman = queen."
Distributed word vectors are powerful and can be used for many applications, particularly word prediction and translation.
Here, we will try to apply them to sentiment analysis.

Recent work out of Stanford has also applied deep learning to sentiment analysis; their code is available in Java. However,
their approach, which relies on sentence parsing, cannot be applied in a straightforward way to paragraphs of arbitrary length.

PYTHON:
Word2vec from GENSIM package. Although Word2Vec does not require graphics processing units (GPUs) like many deep learning
algorithms, it is compute intensive. Both Google's version and the Python version rely on multi-threading
(running multiple processes in parallel on your computer to save time). ln order to train your model in a reasonable amount
of time, you need cython. Word2Vec will run without cython installed, but it will take days to run instead of minutes.
"""

import pandas as pd

# read data
train = pd.read_csv('/Users/myron/Documents/general_stuff/Git/sample_projects/data/imdb/labeledTrainData.tsv',
                    sep='\t', header=0)
test = pd.read_csv('/Users/myron/Documents/general_stuff/Git/sample_projects/data/imdb/testData.tsv',
                    sep='\t', header=0)
unlabelled_train = pd.read_csv('/Users/myron/Documents/general_stuff/Git/sample_projects/data/imdb/unlabeledTrainData.tsv',
                    sep='\t', header=0)