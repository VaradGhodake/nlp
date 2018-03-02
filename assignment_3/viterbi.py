import numpy as np


state_indices = {'N': 0, 'V': 1, 'A': 2, 'P': 3, 'start': -1}
reverse_idx = dict([x[::-1] for x in state_indices.items()])

transition = np.array(
    [[0.13, 0.43, 1e-4, 0.44],
     [0.35, 1e-4, 0.65, 1e-4],
     [1.00, 1e-4, 1e-4, 1e-4],
     [0.26, 1e-4, 0.74, 1e-4],
     [0.29, 1e-4, 0.71, 1e-4]])

prior = {
    'flies':   np.array([0.025, 0.076, 1e-4, 1e-4]),
    'like' :   np.array([0.012, 0.1,   1e-4, 0.068]),
    'the':     np.array([1e-4,  1e-4,  0.54, 1e-4]),
    'a':       np.array([1e-4,  1e-4,  0.36, 1e-4]),
    'flowers': np.array([0.05,  1e-4,  1e-4, 1e-4]),
    'birds':   np.array([0.076, 1e-4,  1e-4, 1e-4]),
    }


def viterbi(sentence, prior, transition):
    trellis = np.zeros(shape=(len(sentence), transition.shape[1]))
    backtrack = np.zeros(shape=trellis.shape, dtype=int)
    backtrack[0, :] = -1
    trellis[0] = transition[state_indices['start']] * prior[sentence[0]]
    for word_i in range(1, len(sentence)):
        for pos_i in range(len(state_indices) - 1):
            probabilites = (prior[sentence[word_i]] *
                            transition[:-1, pos_i] *
                            trellis[word_i - 1, :])
            max_prob_i = np.argmax(probabilites)
            trellis[word_i, pos_i] = probabilites[max_prob_i]
            backtrack[word_i, pos_i] = max_prob_i
    return trellis, backtrack

def viterbi_backtrack(trellis, backtrack):
    l = []
    i = np.argmax(trellis[-1, :])
    l.append(i)
    for word_i in range(trellis.shape[0] - 1, -1, -1):
        l.append(backtrack[word_i, i])
        i = l[-1]
    return l[::-1]

import pprint
pprint.pprint(state_indices)
trellis, backtrack = viterbi("flies like flowers".split(), prior, transition)
pprint.pprint(trellis)
pprint.pprint(backtrack)
backtrack = viterbi_backtrack(trellis, backtrack)
pprint.pprint([reverse_idx[pos] for pos in backtrack])
