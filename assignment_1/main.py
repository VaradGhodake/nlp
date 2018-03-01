import nltk
import os
import numpy as np
import pickle
import pprint
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


os.environ['CLASSPATH'] = os.path.abspath('../stanford-postagger-2017-06-09')

stanford_model = '../stanford-postagger-2017-06-09/models/english-left3words-distsim.tagger'

CRF_model = './crfmodel.bin'
BRILL_model = './brillmodel.bin'
BIGRAM_model = './bgrammodel.bin'
PERCEPTRON_model = './perceptronmodel.pickle'

OUTPUT = './out.bin'

st_tagger = nltk.tag.stanford.StanfordPOSTagger(stanford_model)

brown = nltk.corpus.brown
brown.ensure_loaded()
log.info("Loaded brown corpus")

n_sents = len(brown.sents())

train = int(n_sents * 0.8)
test = n_sents - train

sents = brown.tagged_sents(tagset='universal')
train_set = sents[:train]
test_set = sents[-test:]
log.info("Obtained training and test sets")


crf_tagger = nltk.tag.CRFTagger(verbose=True)
if os.path.exists(CRF_model):
    crf_tagger.set_model_file(CRF_model)
    log.info("Loaded existing CRF model")
else:
    log.info("Training CRF model on train_set of size %d", len(train_set))
    crf_tagger.train(train_set, CRF_model)
    log.info("Trained CRF model")

treebank_to_universal = nltk.tag.mapping.tagset_mapping('en-ptb', 'universal')


def map_st_output(sentences):
    return [[(word[0], treebank_to_universal[word[1]]) for word in sent]
            for sent in sentences]

init_brill_tagger = crf_tagger

if os.path.exists(BRILL_model):
    with open(BRILL_model, 'rb') as f:
        brill_tagger = nltk.tag.brill.BrillTagger.decode_json_obj(
            (init_brill_tagger, ) + pickle.loads(f.read()))
    log.info("Loaded BRILL model")
else:
    brill_templates = nltk.tag.brill.brill24()
    brill = nltk.tag.brill_trainer.BrillTaggerTrainer(
        init_brill_tagger, templates=brill_templates, trace=6)
    log.info("Training brill model")
    brill_tagger = brill.train(train_set)
    with open(BRILL_model, 'wb') as f:
        f.write(pickle.dumps(brill_tagger.encode_json_obj()[1:]))
    log.info("Saved brill model")

fallback_tagger = nltk.tag.DefaultTagger('NOUN')
if os.path.exists(BIGRAM_model):
    with open(BIGRAM_model, 'rb') as f:
        bgram_tagger = nltk.tag.sequential.BigramTagger.decode_json_obj(
            (pickle.loads(f.read()), fallback_tagger))
    log.info("Loaded BIGRAM model")
else:
    log.info("Training bgram model")
    bgram_tagger = nltk.tag.sequential.BigramTagger(
        train=train_set, backoff=fallback_tagger, verbose=True)
    with open(BIGRAM_model, 'wb') as f:
        f.write(pickle.dumps(bgram_tagger.encode_json_obj()[0]))
    log.info("Saved BIGRAM model")

if os.path.exists(PERCEPTRON_model):
    perceptron_tagger = nltk.tag.perceptron.PerceptronTagger(load=False)
    perceptron_tagger.load(PERCEPTRON_model)
    log.info("Loaded Perceptron model")
else:
    log.info("Training perceptron model")
    perceptron_tagger = nltk.tag.perceptron.PerceptronTagger(load=False)
    perceptron_tagger.train(sentences=train_set, save_loc=PERCEPTRON_model)
    log.info("Saved Perceptron model")

# if not os.path.exists(OUTPUT):
if True:
    raw_sents = [[x[0] for x in sent] for sent in test_set]
    import pdb
    pdb.set_trace()
    raw_sents = ["I went to the bank to withdraw some money".split(),
                 "I went to the bank with my friends".split(),
                 "I went to the bank near the bank to ".split(),
                 "I bank on the bank which is near the bank".split()]
    raw_tags = [[x[1] for x in sent] for sent in test_set]
    log.info("Split raw sentences and tags")
    st_out = map_st_output(st_tagger.tag_sents(raw_sents))
    pprint.pprint(st_out)
    st_tags = [[x[1] for x in sent] for sent in st_out]
    log.info("Got stanford output")
    crf_out = crf_tagger.tag_sents(raw_sents)
    pprint.pprint(crf_out)
    crf_tags = [[x[1] for x in sent] for sent in crf_out]
    log.info("Got crf output")
    brill_out = brill_tagger.tag_sents(raw_sents)
    pprint.pprint(brill_out)
    brill_tags = [[x[1] for x in sent] for sent in brill_out]
    log.info("Got brill output")
    bgram_out = bgram_tagger.tag_sents(raw_sents)
    pprint.pprint(bgram_out)
    bgram_tags = [[x[1] for x in sent] for sent in bgram_out]
    log.info("Got bgram output")
    perceptron_out = perceptron_tagger.tag_sents(raw_sents)
    pprint.pprint(perceptron_out)
    perceptron_tags = [[x[1] for x in sent] for sent in perceptron_out]
    log.info("Got perceptron output")
    import pdb
    pdb.set_trace()

    with open(OUTPUT, 'wb') as f:
        pickle.dump({
            'raw': raw_sents,
            'tags': raw_tags,
            'st': st_out,
            'st_tags': st_tags,
            'crf': crf_out,
            'crf_tags': crf_tags,
            'brill': brill_out,
            'brill_tags': brill_tags,
            'bgram': bgram_out,
            'bgram_tags': bgram_tags,
            'perceptron': perceptron_out,
            'perceptron_tags': perceptron_tags}, f)
        log.info("Saving all outputs")
else:
    with open(OUTPUT, 'rb') as f:
        output = pickle.load(f)


stats = dict()


def flatten(l):
    return [x for y in l for x in y]

z = zip(flatten(output['tags']),
        flatten(output['st_tags']),
        flatten(output['crf_tags']),
        flatten(output['brill_tags']),
        flatten(output['bgram_tags']),
        flatten(output['perceptron_tags']))

tag_order = dict([(y, x) for x, y in
                  enumerate(nltk.tag.mapping._UNIVERSAL_TAGS)])
tag_order[None] = tag_order['NOUN']


stats['st'] = np.zeros(shape=(len(tag_order), len(tag_order)), dtype=int)
stats['crf'] = np.zeros(shape=(len(tag_order), len(tag_order)), dtype=int)
stats['brill'] = np.zeros(shape=(len(tag_order), len(tag_order)), dtype=int)
stats['bgram'] = np.zeros(shape=(len(tag_order), len(tag_order)), dtype=int)
stats['perceptron'] = np.zeros(shape=(len(tag_order), len(tag_order)),
                               dtype=int)


print("Considering real (brown corpus) tags instead of stanford because "
      "they will be more reliable, and because stanford tagger is not "
      "trained on 80% brown corpus like the others")
for real, st, crf, brill, bgram, perceptron in z:
    stats['st'][tag_order[st], tag_order[real]] += 1
    stats['crf'][tag_order[crf], tag_order[real]] += 1
    stats['brill'][tag_order[brill], tag_order[real]] += 1
    stats['bgram'][tag_order[bgram], tag_order[real]] += 1
    stats['perceptron'][tag_order[perceptron], tag_order[real]] += 1

np.set_printoptions(linewidth=200)
for k, v in stats.items():
    print(k, '-'*80)
    print(nltk.tag.mapping._UNIVERSAL_TAGS)
    print(v)


def table_of_confusion(mat, i):
    # mat is a confusion matrix of all classes of the form:
    # p       real class
    # r       0  i  2  3
    # e c   0 tn|fn|tn tn
    # d l    ---+--+-----
    # i a   i fp|TP|fp fp
    # c s    ---+--+-----
    # t s   2 tn|fn|tn tn
    # e     3 tn|fn|tn tn
    # d

    true_positive = mat[i, i]
    false_negative = np.sum(mat[:, i]) - true_positive
    false_positive = np.sum(mat[i, :]) - true_positive
    true_negative = np.sum(mat) - (true_positive +
                                   false_negative + false_positive)

    return {'tp': true_positive, 'tn': true_negative,
            'fp': false_positive, 'fn': false_negative}

print('-' * 80)
for k, m in stats.items():
    print("Performance of", k, "tagger")
    for i, tag in enumerate(nltk.tag.mapping._UNIVERSAL_TAGS):
        print("For", tag)
        toc = table_of_confusion(m, i)
        print(toc)
        accuracy = ((toc['tp'] + toc['tn']) /
                    (toc['tp'] + toc['tn'] + toc['fp'] + toc['fn']))
        recall = (toc['tp']) / (toc['tp'] + toc['fn'])
        precision = (toc['tp']) / (toc['tp'] + toc['fp'])
        f1 = 2 / ((1 / recall) + (1 / precision))
        print("Accuracy:", accuracy,
              "precision:", precision,
              "recall:", recall,
              "f1:", f1)
    print('-' * 80)
