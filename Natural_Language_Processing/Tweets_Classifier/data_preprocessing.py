import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import utils

if __name__ == '__main__':

    DATASET_SIZE = 10000
    TEST_RATIO = 0.25
    SEED = 1

    np.random.seed(seed=SEED)

    fifa_tweets = pd.read_csv("data/raw/FIFA.csv", dtype=str)['Orig_Tweet']
    fifa_tweets = np.random.choice(fifa_tweets, DATASET_SIZE // 2)
    fifa_tweets = np.expand_dims(fifa_tweets, 1)
    fifa_tweets = np.concatenate((fifa_tweets, np.zeros(fifa_tweets.shape).astype(int)), axis=1)

    political_tweets = pd.read_csv("data/raw/pol_tweets.csv", dtype=str)['tweet_text']
    political_tweets = np.random.choice(political_tweets, DATASET_SIZE // 2)
    political_tweets = np.expand_dims(political_tweets, 1)
    political_tweets = np.concatenate((political_tweets, np.ones(political_tweets.shape).astype(int)), axis=1)

    ds = np.concatenate((fifa_tweets, political_tweets), axis=0)
    train_set, test_set = train_test_split(ds, test_size=TEST_RATIO, shuffle=True)


    nlp = spacy.load('en')  # to be used in PreProcess()

    words_to_remove = ['-PRON-']
    training_corpus = utils.PreProcess(train_set[:, 0], nlp, words_to_remove=words_to_remove)
    np.save("data/training_corpus.npy", training_corpus)

    keys = utils.Find_Keys(training_corpus)
    np.save("data/keys.npy", keys)

    X_train = utils.Corpus2Vectors(training_corpus, keys)
    y_train = train_set[:, 1].astype(int)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)

    test_corpus = utils.PreProcess(test_set[:, 0], nlp, words_to_remove=words_to_remove)
    X_test = utils.Corpus2Vectors(test_corpus, keys)
    y_test = test_set[:, 1].astype(int)
    np.save("data/test_corpus.npy", test_corpus)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)









