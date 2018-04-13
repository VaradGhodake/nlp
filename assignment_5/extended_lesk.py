import nltk
wn = nltk.wordnet.wordnet
wn.ensure_loaded()

import difflib


def get_gloss(synset):
    return nltk.word_tokenize(
        '\n'.join([synset.definition()] + synset.examples()))


def get_context(tokens, target_idx):
    if target_idx < 2:
        return tokens[:5], target_idx
    elif target_idx >= len(tokens) - 2:
        r = tokens[-5:]
        return r, len(r) - (len(tokens) - target_idx)
    else:
        return tokens[target_idx - 2:target_idx + 3], 2

def comb(seq):
    if len(seq):
        for sense in seq[0]:
            for suffix in comb(seq[1:]):
                yield [sense] + list(suffix)
    else:
        yield []


def overlap(gloss1, gloss2):
    differ = difflib.SequenceMatcher(None, gloss1, gloss2, autojunk=False)
    return sum(match.size**2 for match in differ.get_matching_blocks())


def compare(combination, func):
    s = 0
    for i, w1 in enumerate(combination):
        for j, w2 in enumerate(combination[i + 1:]):
            s += func(get_gloss(w1), get_gloss(w2))
    return s


def score(combination):
    return compare(combination, overlap)


def extended_lesk(sent, idx):
    sent = nltk.word_tokenize(sent)
    sent, idx = get_context(sent, idx)
    senses = [wn.synsets(word) for word in sent]
    senses = [(i, x) for i, x in enumerate(senses) if x]
    context = [sent[i] for i, _ in senses]
    senses = [x[1] for x in senses]
    combinations = list(comb(senses))
    scores = [score(c) for c in combinations]
    select = scores.index(max(scores))
    return combinations[select][idx]


if __name__ == '__main__':
    ans = extended_lesk('one two three four five', 3)
    print(ans)
    print(ans.definition())

