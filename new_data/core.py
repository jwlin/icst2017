"""
Core functions to train and test data with different techniques (pattern-based, similarity-based, ML-based)
"""

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

from bs4 import BeautifulSoup
import random
#import spacy
#from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from gensim import corpora, models, similarities
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


# local import
from util import is_validated

#nlp = spacy.load('en_core_web_sm')
#stemmer = nltk.stem.SnowballStemmer('english')
custom_stopwords = ['your a the is and or in be to of for not on with as by']


def train_by_pattern(x_train):
    pattern_to_topic = {}
    for d in x_train:
        if d['pattern'] in pattern_to_topic:
            pattern_to_topic[d['pattern']].add(d['topic'])
        else:
            pattern_to_topic[d['pattern']] = set([d['topic']])
    return pattern_to_topic


def pred_by_pattern(x_test, pattern_to_topic):
    y_pred = []
    for d in x_test:
        input_tag = BeautifulSoup(d['dom'], 'html5lib').find('input')
        ans = set()
        #patt = set()
        for pattern in pattern_to_topic.keys():
            if is_validated(input_tag, pattern):
                #patt.add(pattern)
                ans = ans.union(pattern_to_topic[pattern])
        '''
        if len(ans) > 1:
            print(d['dom'])
            print(d['id'], len(ans), ans)
            print(d['pattern'])
            for a in ans:
                print([(k, v) for k, v in pattern_to_topic.items() if k in patt and a in v])
            input()
        '''
        if not ans:
            y_pred.append('UNK')
        else:
            y_pred.append(random.sample(ans, 1)[0])
    return y_pred


def pred_by_pattern_sim(x_test, pattern_to_topic, y_pred_sim, mode):
    # mode: {'no-match', 'multiple', 'both'}
    y_pred = []
    for idx, d in enumerate(x_test):
        input_tag = BeautifulSoup(d['dom'], 'html5lib').find('input')
        ans = set()
        for pattern in pattern_to_topic.keys():
            if is_validated(input_tag, pattern):
                ans = ans.union(pattern_to_topic[pattern])
        if not ans:
            if mode in ['no-match', 'both']:
                y_pred.append(y_pred_sim[idx])
            else:
                y_pred.append('UNK')
        elif len(ans) == 1:
            y_pred.append(list(ans)[0])
        else:
            if mode in ['multiple', 'both']:
                if y_pred_sim[idx] in ans:
                    y_pred.append(y_pred_sim[idx])
                else:
                    ans.add(y_pred_sim[idx])
                    y_pred.append(random.sample(ans, 1)[0])
            else:
                y_pred.append(random.sample(ans, 1)[0])
    return y_pred


def train_by_sim(x_train):
    # build dictionary
    corpus = []
    for d in x_train:
        #corpus.append([w.lemma_ for w in nlp(d['feature'])])  # lemmatize
        #corpus.append([stemmer.stem(w) for w in d['feature'].split()])  # stemming
        corpus.append(d['feature'].split())
    # print(corpus)
    dictionary = corpora.Dictionary(corpus)
    stoplist = set(custom_stopwords)
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    #once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    once_ids = []
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()

    # build, transfrom, and index corpus
    corpus_bow = []
    for d in x_train:
        #corpus_bow.append(dictionary.doc2bow([w.lemma_ for w in nlp(d['feature'])]))  # lemmatize
        #corpus_bow.append(dictionary.doc2bow([stemmer.stem(w) for w in d['feature'].split()]))  # stemming
        corpus_bow.append(dictionary.doc2bow(d['feature'].split()))
    #print(corpus_bow)
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    #print(corpus_tfidf)
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
    corpus_lsi = lsi[corpus_tfidf]
    #print(corpus_lsi)
    index = similarities.MatrixSimilarity(corpus_lsi)
    idx_to_topic = {idx: d['topic'] for idx, d in enumerate(x_train)}
    assert len(idx_to_topic) == len(corpus_bow)  # make sure all docs are transformed and the index is correct
    return dictionary, tfidf, lsi, index, idx_to_topic


def pred_by_sim(x_test, dictionary, tfidf, lsi, index, idx_to_topic):
    y_pred = []
    for d in x_test:
        #vec = [w.lemma_ for w in nlp(d['feature'])]  # lemmatize
        #vec = [stemmer.stem(w) for w in d['feature'].split()]  # stemming
        vec = d['feature'].split()
        vec_bow = dictionary.doc2bow(vec)
        vec_tfidf = tfidf[vec_bow]
        vec_lsi = lsi[vec_tfidf]
        sims = index[vec_lsi]
        # sims = [(166, 0.9999132), (666, 0.9885738), ...]
        sims = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
        #print('Feature', d['feature'])
        #print('Ans:', d['topic'])
        #for idx, sim_score in sims[:5]:
        #    print(idx_to_topic[idx], sim_score)
        ans = idx_to_topic[sims[0][0]]
        #if (sims[0][1] - sims[4][1]) < 0.1:  # the similarity range of top 5 items is less than 0.1
        #    ans = vote([idx_to_topic[s[0]] for s in sims[:5]])
        #print('inferred ans:', ans)
        #input()
        y_pred.append(ans)
    return y_pred


def vote(topics):
    t_to_c = defaultdict(int)
    for t in topics:
        t_to_c[t] += 1
    return sorted(t_to_c.items(), key=lambda x: x[1], reverse=True)[0][0]


def train_by_ml(x_train, model='nb'):
    corpus = []
    y_train = []
    for d in x_train:
        corpus.append(d['feature'])
        y_train.append(d['topic'])

    # model: ['nb', 'svm', 'logit', rf']
    if model == 'nb':
        classifier = Pipeline([('vect', CountVectorizer(stop_words=custom_stopwords, ngram_range=(1, 1))),
                               #('tfidf', TfidfTransformer()),
                               ('chi2', SelectKBest(chi2, k=1000)),
                               ('clf', MultinomialNB(alpha=0.1))
                        ])
    elif model == 'svm':
        classifier = Pipeline([('vect', CountVectorizer(stop_words=custom_stopwords, ngram_range=(1, 3))),
                               ('tfidf', TfidfTransformer()),
                               ('clf', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1e-4, max_iter=1000, tol=1e-3))])
    elif model == 'logit':
        classifier = Pipeline([('vect', CountVectorizer(stop_words=custom_stopwords, ngram_range=(1, 1))),
                               ('tfidf', TfidfTransformer(use_idf=False)),
                               ('clf', SGDClassifier(loss='log', penalty='l1', alpha=1e-5, max_iter=1000, tol=1e-3))])
    elif model == 'rf':
        classifier = Pipeline([('vect', CountVectorizer(stop_words=custom_stopwords, ngram_range=(1, 1))),
                               ('clf', RandomForestClassifier(criterion='gini', min_samples_leaf=1,
                                                              min_samples_split=2, n_estimators=50,
                                                              oob_score=True, n_jobs=-1))])
    else:
        assert False, 'Wrong ML model'
    return classifier.fit(corpus, y_train)


def pred_by_ml(classifier, x_test):
    corpus = []
    for d in x_test:
        corpus.append(d['feature'])
    return classifier.predict(corpus)