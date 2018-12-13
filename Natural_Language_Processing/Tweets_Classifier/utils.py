import numpy as np


def GoodToken(token):
    unwanted_strings = ['\\', 'rt', '@', '\r', '\n', '\t']
    unwanted = False
    for string in unwanted_strings:
        if string in str(token).lower():
            unwanted = True
            break
    return not (token.is_stop or token.is_punct or
                token.like_num or token.like_url or unwanted)


def Refine(tweet):
    return tweet.replace("#", "")


def PreProc_tweet(tweet, nlp):
    text = nlp(Refine(tweet))
    bag = [token.lemma_ for token in text if GoodToken(token)]
    return bag


def RemoveWords(corpus, list_of_words):
    for word in list_of_words:
        for i in range(len(corpus)):
            if word in corpus[i]:
                corpus[i].remove(word)


def PreProcess(set_of_tweets, nlp, words_to_remove=[], verbose_step=1000):
    set_of_bags = []
    num_tweets = len(set_of_tweets)

    for idx in range(num_tweets):
        tweet = set_of_tweets[idx]
        set_of_bags += [PreProc_tweet(tweet, nlp)]
        if idx % verbose_step == 0:
            print('{}/{} tweets processed\n'.format(idx, num_tweets))

    RemoveWords(set_of_bags, words_to_remove)
    return set_of_bags


def Add_Keys(keys, bag):

    for key in bag:
        if key in keys:
            continue
        else:
            keys.append(key)


def Find_Keys(corpus):
    keys = []
    for bag in corpus:
        Add_Keys(keys, bag)
    return keys


def Bag2Vec(bag, keys):
    num_keys = len(keys)
    vec = np.zeros(num_keys)
    for idx in range(num_keys):
        vec[idx] = (int(keys[idx] in bag))
    return vec


def Corpus2Vectors(corpus, keys):
    num_keys = len(keys)
    num_bags = len(corpus)
    X_vec = np.zeros((num_bags, num_keys))

    for idx in range(num_bags):
        X_vec[idx] = Bag2Vec(corpus[idx], keys)
    return X_vec


def AssessModel(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix
    tot = y_test.shape[0]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = np.trace(cm)/tot
    print(cm)
    print('Accuracy: {}'.format(acc))



