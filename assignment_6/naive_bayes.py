#!/usr/bin/env python
import os
import os.path
import nltk
import dill as pickle

from collections import defaultdict
from collections import deque

import semcor_reader

wn = nltk.wordnet.wordnet
wn.ensure_loaded()


def get_corpus_stats(folder):
    prior_prob = {}
    sense_count = defaultdict(int)
    feature_prob = [defaultdict(lambda: defaultdict(int)) for _ in range(4)]
    colocation = deque([None] * 3, maxlen=3)
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        print(f)
        output = semcor_reader.read_semcor(open(f))
        for para in output:
            for sent in para:
                for token in sent:
                    lemma = token['lemma']
                    senses = token['true_senses']
                    for word in token['words']:
                        colocation.append(word)
                    if lemma is not None and senses:
                        senses = [x.key() if not isinstance(x, str) else x for x in senses]
                        word = token['words'][0]
                        if word not in prior_prob:
                            prior_prob[word] = defaultdict(int)

                        features = list(colocation)
                        features.append(token['pos'])
                        for sense in senses:
                            prior_prob[word][sense] += 1
                            sense_count[sense] += 1
                        for i, feature in enumerate(features):
                            for sense in senses:
                                feature_prob[i][feature][sense] += 1
    return sense_count, prior_prob, feature_prob


def calc_prior(sense, word, prior_dict):
    return prior_prob[word][sense] / sum(prior_prob[word].values())


def disambiguate(tokens, index, sense_count, prior_prob, feature_prob):
    colocation = deque([None] * 3, maxlen=3)
    for word, pos in tokens[index - 2: index + 1]:
        colocation.append(word)
    feature_vec = list(colocation) + [tokens[index][1]]

    args = []
    word = tokens[index][0]
    lemmas = wn.lemmas(word)
    for sense in lemmas:
        sense = sense.key()
        prior = prior_prob[word][sense] / sum(prior_prob[word].values())
        feature = 1
        for f, this_f in zip(feature_prob, feature_vec):
            try:
                feature *= f[this_f][sense] / sense_count[sense]
            except ZeroDivisionError:
                pass
        args.append(prior * feature)
    return lemmas[args.index(max(args))]

if __name__ == '__main__':
    F = 'tmp.pickle'
    if os.path.exists(F):
        with open(F, 'rb') as f:
            sense_count, prior_prob, feature_prob = pickle.load(f)
    else:
        sense_count, prior_prob, feature_prob = get_corpus_stats("./data/brown1/tagfiles")
        with open(F, 'wb') as f:
            pickle.dump((sense_count, prior_prob, feature_prob), f)
    print(disambiguate([('He', 'p'), ('deposited', 'v'), ('in', 'x'), ('the', 'a'),
                        ('bank', 'n')], 4, sense_count, prior_prob,
                        feature_prob))






