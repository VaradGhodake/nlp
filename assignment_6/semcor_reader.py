#!/usr/bin/env python
import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

import nltk
import bs4
wordnet = nltk.wordnet.wordnet
wordnet.ensure_loaded()


def penn2morphy(penntag):
    morphy_tag = {'NN':wordnet.NOUN, 'JJ':wordnet.ADJ,
                  'VB':wordnet.VERB, 'RB':wordnet.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except KeyError:
        return None


def read_para(para):
    sents = []
    for sentence in para.findAll('s'):
        tokens = []
        for token in  sentence.children:
            if token == '\n':
                continue
            if token.get('lemma') is not None:
                disambiguate = True
                lemma = token['lemma']
                lexsn = token['lexsn']
                pos = token['pos'].upper()
                pos = penn2morphy(pos)
                lexsn_s = lexsn.split(';')
                lemma_keys = [lemma + '%' + lexsn for lexsn in lexsn_s]
                wn_lemmas = []
                for lemma_key in lemma_keys:
                    try:
                        wn_lemma = wordnet.lemma_from_key(lemma_key)
                    except nltk.corpus.reader.wordnet.WordNetError:
                        log.warning("No wordnet lexsn found for %s", lemma_key)
                        wn_lemma = lemma_key
                    wn_lemmas.append(wn_lemma)
            else:
                disambiguate = False
                wn_lemmas = None
                lemma = None
                pos = None
            string = token.string
            words = nltk.tokenize.word_tokenize(string.replace('_', ' '))
            tokens.append({'words': words,
                           'disambiguate?': disambiguate,
                           'true_senses': wn_lemmas,
                           'lemma': lemma,
                           'pos': pos})
        else:
            sents.append(tokens)
    return sents


def read_semcor(f):
    soup = bs4.BeautifulSoup(f.read(), 'html.parser')
    context = soup.find('context')
    output = []
    for para in context.findAll('p'):
        sents = read_para(para)
        output.append(sents)
    return output


if __name__ == '__main__':
    output = read_semcor(open('../datasets/semcor3.0/brownv/tagfiles/br-a06'))
    import pdb
    pdb.set_trace()
    output = output

