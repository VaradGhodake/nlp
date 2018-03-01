import numpy as np


def levenshtein(s0, s1):
    m = np.zeros((len(s1) + 1, len(s0) + 1), dtype=int)
    m[0, :] = range(len(s0) + 1)
    m[:, 0] = range(len(s1) + 1)

    for i1 in range(1, len(s1) + 1):
        for i0 in range(1, len(s0) + 1):
            m[i1, i0] = min(m[i1 - 1, i0] + 1,
                            m[i1, i0 - 1] + 1,
                            m[i1 - 1, i0 - 1] +
                            (0 if s0[i0 - 1] == s1[i1 - 1] else 1))
    print(m)
    return m[-1, -1]

print(levenshtein('execution', 'intention'))
