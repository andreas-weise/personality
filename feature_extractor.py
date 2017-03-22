import nltk
import numpy
import re
from nltk.util import ngrams
import operator
import gensim
from collections import defaultdict
from math import sqrt
import json
import requests
import hashlib
import time
from operator import itemgetter
import pickle
import os

# extract configured set of features from list of text instances
# global variables to pass around data
source_texts = []
tokenized_texts_case = []
word_counts = []
tokenized_texts = []
tagged_texts = []
cropped_texts = []
stemmed_texts = []
stemmed_cropped_texts = []
w2v_model = None


def preprocess():
    """ prepares source_texts for feature extraction; called by extract_features

    puts words in lower case, tokenizes and stems them, and removes rare words
    no args and no return because of use of global variables
    """

    global source_texts, tokenized_texts_case, word_counts, tokenized_texts, \
        tagged_texts, cropped_texts, stemmed_texts, stemmed_cropped_texts

    # load the processed texts from pickle dumps if those exist (check for one)
    if os.path.exists('pickle/tokenized_texts_case.p'):
        with open('pickle/tokenized_texts_case.p', 'rb') as p_file:
            tokenized_texts_case = pickle.load(p_file)
        with open('pickle/word_counts.p', 'rb') as p_file:
            word_counts = pickle.load(p_file)
        with open('pickle/tokenized_texts.p', 'rb') as p_file:
            tokenized_texts = pickle.load(p_file)
        with open('pickle/tagged_texts.p', 'rb') as p_file:
            tagged_texts = pickle.load(p_file)
        with open('pickle/cropped_texts.p', 'rb') as p_file:
            cropped_texts = pickle.load(p_file)
        with open('pickle/stemmed_texts.p', 'rb') as p_file:
            stemmed_texts = pickle.load(p_file)
        with open('pickle/stemmed_cropped_texts.p', 'rb') as p_file:
            stemmed_cropped_texts = pickle.load(p_file)
        return

    # lower case, count words, tokenize, and tag
    tokenized_texts_case = [nltk.word_tokenize(text) for text in source_texts]
    source_texts = [text.lower() for text in source_texts]
    word_counts = [len(text.split()) for text in source_texts]
    tokenized_texts = [nltk.word_tokenize(text) for text in source_texts]
    tagged_texts = [[tag[1] for tag in nltk.pos_tag(text)]
                    for text in tokenized_texts]

    stop_list = nltk.corpus.stopwords.words('english')
    stop_list.extend(['.', ',', ':', ';', '(', ')', '!', '?', '"', "'", "''",
                      '``', '-', "'s", 'would', '[', ']', '{', '}', '...',
                      'p.'])
    cropped_texts = [[word for word in text if word not in stop_list]
                     for text in tokenized_texts]

    # stem using standard nltk porter stemmer
    porter = nltk.PorterStemmer()
    # stemmed_texts = [[porter.stem(t) for t in tokens]
    #                 for tokens in tokenized_texts]
    # iterating instead of list comprehension to allow exception handling
    for tokens in tokenized_texts:
        stemmed_text = []
        for t in tokens:
            try:
                stemmed_text.extend([porter.stem(t)])
            except IndexError:
                stemmed_text.extend('')
        stemmed_texts.append(stemmed_text)
    for tokens in cropped_texts:
        stemmed_cropped_text = []
        for t in tokens:
            try:
                stemmed_cropped_text.extend([porter.stem(t)])
            except IndexError:
                stemmed_cropped_text.extend('')
        stemmed_cropped_texts.append(stemmed_cropped_text)

    # remove rare words
    # vocab = nltk.FreqDist(w for w in line for line in stemmed_texts)
    vocab = nltk.FreqDist(w for text in stemmed_texts for w in text)
    rare_words_list = [re.escape(word) for word in vocab.hapaxes()]
    rare_words_regex = re.compile(r'\b(%s)\b' % '|'.join(rare_words_list))
    stemmed_texts = [[rare_words_regex.sub('<RARE>', w) for w in text]
                     for text in stemmed_texts]
    # note: source_texts will be lower case, but only stemmed_texts will have
    # rare words removed

    # dump the processed texts to pickle files for next time they are needed
    with open('pickle/tokenized_texts_case.p', 'wb') as p_file:
        pickle.dump(tokenized_texts_case, p_file)
    with open('pickle/word_counts.p', 'wb') as p_file:
        pickle.dump(word_counts, p_file)
    with open('pickle/tokenized_texts.p', 'wb') as p_file:
        pickle.dump(tokenized_texts, p_file)
    with open('pickle/tagged_texts.p', 'wb') as p_file:
        pickle.dump(tagged_texts, p_file)
    with open('pickle/cropped_texts.p', 'wb') as p_file:
        pickle.dump(cropped_texts, p_file)
    with open('pickle/stemmed_texts.p', 'wb') as p_file:
        pickle.dump(stemmed_texts, p_file)
    with open('pickle/stemmed_cropped_texts.p', 'wb') as p_file:
        pickle.dump(stemmed_cropped_texts, p_file)


def bag_of_function_words():
    """ returns, for each nltk stop word, count per text in source_texts """
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, text))
                  for text in source_texts]
        counts = [counts[i] / word_counts[i] for i in range(0, len(counts))]
        bow.append(counts)
    return bow


def bag_of_ngrams(texts, n=1, m=None):
    """ returns counts of up to m overall most common ngrams for each given text

    determines the counts of all ngrams, orders them by sum of counts across
    texts and returns counts for up to m most common ones

    args:
        texts: list of texts as list of list of words (or tags etc)
        n: 1 for unigram (default), 2 for bigram, 3 for trigram etc.
        m: upper limit for number of features; if none, all are returned

    returns:
        list of list of most common ngram counts, m x len(texts)
    """
    # generate list of lists of ngrams for all texts
    ngrammed_texts = [list(ngrams(text, n)) for text in texts]

    # count ngrams in dictionaries, one for each text, plus one for sums
    cnts = []
    cnt_sum = defaultdict(int)
    for text in ngrammed_texts:
        cnts.append(defaultdict(int))
        i = len(cnts) - 1
        for ngram in text:
            cnts[i][ngram] += 1
            cnt_sum[ngram] += 1

    # create list of lists of counts for each text for the most common ngrams
    # first, sort the ngrams by total counts
    cnt_sorted = sorted(cnt_sum.items(), key=operator.itemgetter(1),
                        reverse=True)
    # then, create the bag of ngrams (up to m), normalized by word count
    bon = []
    for ngram, total in cnt_sorted:
        counts = [cnt[ngram] for cnt in cnts]
        counts = [counts[i] / word_counts[i] for i in range(0, len(counts))]
        bon.append(counts)
        if m and len(bon) >= m:
            break
    return bon


# def bag_of_char_ngrams(texts, n=1, m=None):
#     """ returns counts of up to m overall most common character ngrams for
# each given text"""


def unique_words_ratio():
    """ returns #unique words / #words for each text

    uses stemmed words so 'eat' and 'eating' etc. are not treated as distinct
    (assuming they are stemmed correctly; 'eat' and 'ate' are still 'distinct');
    note that punctuation characters, parentheses etc. are treated as words
    """
    return [[len(set(text)) / len(text) for text in stemmed_texts]]


def words_per_sentence():
    """ returns average number of words per sentence for each text

    uses the '.' POS tag to detect number of sentences to avoid treating '.' in
    abbreviations as sentence ends
    """
    return [[word_counts[i] /
             (tagged_texts[i].count('.') if tagged_texts[i].count('.') > 0
              else 1)
             for i in range(0, len(word_counts))]]


def characters_per_words():
    """ returns average number of characters per word for each text

    note that character count includes punctuation, parentheses etc.
    """
    return [[(len(source_texts[i]) - word_counts[i] + 1) / word_counts[i]
             for i in range(0, len(word_counts))]]


def topic_model_scores(num_topics):
    """ returns, for the top num_topics topics (lsi), the score for each text

    args:
        num_topics: number of topics (features) to consider
    """
    dictionary = gensim.corpora.Dictionary(cropped_texts)
    corpus = [dictionary.doc2bow(text) for text in cropped_texts]
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary,
                                          num_topics=num_topics)
    corpus_lsi = lsi[corpus_tfidf]

    return [[scores[i][1] for scores in corpus_lsi]
            for i in range(0, num_topics)]


def word2vec_avg():
    """ returns avg vector for words in each text """
    return [[sum(w2v_model[token][i] for token in text if token in w2v_model) /
             len(text)
             # use texts in original case (google word2vec is case sensitive)
             for text in tokenized_texts_case]
            for i in range(0, 50)]


def word2vec_max_val():
    """ returns vector of max value for each dim for all words in each text """
    return [[max(w2v_model[token][i] for token in text if token in w2v_model)
             # use texts in original case (google word2vec is case sensitive)
             for text in tokenized_texts_case]
            for i in range(0, 50)]


def word2vec_avg_max_abs(n=5):
    """ returns avg of n vectors with max abs value for words in each text """
    # compute absolute values for word vectors for words in each text
    abs_vals = [[[w2v_model[token],
                  sqrt(sum(val * val for val in w2v_model[token]))]
                 for token in text if token in w2v_model]
                for text in tokenized_texts_case]
    # sort vectors within texts by absolute values
    abs_vals_sorted = [sorted(vec_lst, key=operator.itemgetter(1), reverse=True)
                       for vec_lst in abs_vals]
    # return average of top n vectors for each text
    return [[sum(vec_lst[j][0][i] for j in range(0, min(n, len(vec_lst)))) /
             min(n, len(vec_lst))
             for vec_lst in abs_vals_sorted]
            for i in range(0, 50)]


def liwc_scores():
    """ returns 93 liwc scores for each text"""
    def get_liwc_scores(text):
        """ aux function to handle the api call to liwc for a single text"""

        api_key = '58d00611e53b0b05af5239d6'
        api_secret = 'isYCnugw39h025UjvQe5ZCdCKhj1EgaAHjZjsIbPips'

        # hash + timestamp as identifer for each text
        # must be unique and texts apparently cannot be deleted after upload
        text_index = '%s_%s' % (
            hashlib.sha1(text.encode()).hexdigest(),
            time.time())

        headers = {
            'X-API-KEY': api_key,
            'X-API-SECRET-KEY': api_secret,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        data = {
            'name': text_index,
            'person_handle': text_index,
            'gender': 0,
            'content': {
                'language_content': text
            }
        }

        response = requests.post('https://app.receptiviti.com/v2/api/person',
                                 headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception('API call for LIWC scores failed!')

        liwc_raw = response.json()['contents'][0]['liwc_scores']
        # 7 keys directly contain a score, 1 key contains dict; flatten this
        liwc_tuples = [(key, val) for (key, val) in liwc_raw.items()
                       if key != 'categories']
        liwc_tuples.extend([(key, val) for (key, val)
                            in liwc_raw['categories'].items()])
        # return just the scores, sorted by their keys
        return [val for (key, val) in sorted(liwc_tuples, key=itemgetter(0))]

    text_scores = [get_liwc_scores(text) for text in source_texts]
    return [[scores[i] for scores in text_scores]
            for i in range(0, len(text_scores[0]))]


def extract_features(texts, conf):
    """ extracts features in given conf from each text in given list of texts

    args:
        texts: list of texts from which to extract features
        conf: set of identifiers of features to be extracted; from conf file

    returns:
        list of lists, #instances x #features = len(texts) x len(conf)
    """
    def load_or_compute(feature):
        feat_data = []
        if os.path.exists('pickle/%s.p' % feature):
            with open('pickle/%s.p' % feature, 'rb') as p_file:
                feat_data = pickle.load(p_file)
        else:
            if feature == 'bag_of_function_words':
                feat_data = bag_of_function_words()
            if feature == 'bag_of_pos_trigrams':
                feat_data = bag_of_ngrams(tagged_texts, 3, 500)
            if feature == 'bag_of_pos_bigrams':
                feat_data = bag_of_ngrams(tagged_texts, 2, 100)
            if feature == 'bag_of_pos_unigrams':
                feat_data = bag_of_ngrams(tagged_texts, 1, None)
            if feature == 'bag_of_trigrams':
                feat_data = bag_of_ngrams(stemmed_texts, 3, 500)
            if feature == 'bag_of_bigrams':
                feat_data = bag_of_ngrams(stemmed_texts, 2, 100)
            if feature == 'bag_of_unigrams':
                feat_data = bag_of_ngrams(stemmed_cropped_texts, 1, 100)
            if feature == 'characters_per_word':
                feat_data = characters_per_words()
            if feature == 'unique_words_ratio':
                feat_data = unique_words_ratio()
            if feature == 'words_per_sentence':
                feat_data = words_per_sentence()
            if feature == 'topic_model_scores':
                feat_data = topic_model_scores(20)
            if feature == 'word2vec_avg':
                feat_data = word2vec_avg()
            if feature == 'word2vec_max_val':
                feat_data = word2vec_max_val()
            if feature == 'word2vec_avg_max_abs':
                feat_data = word2vec_avg_max_abs()
            if feature == 'liwc_scores':
                feat_data = liwc_scores()
            if feature == 'char_unigram':
                feat_data = bag_of_ngrams(source_texts, 1, 100)
            if feature == 'char_bigram':
                feat_data = bag_of_ngrams(source_texts, 2, 100)
            if feature == 'char_trigram':
                feat_data = bag_of_ngrams(source_texts, 3, 500)
            with open('pickle/%s.p' % feature, 'wb') as p_file:
                pickle.dump(feat_data, p_file)
        return feat_data

    all_features = conf is None or len(conf) == 0

    # use global variables to pass around data
    global source_texts
    if len(source_texts) == 0:
        source_texts = texts
        preprocess()
        global w2v_model
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            'data/GoogleNews-vectors-negative300.bin.gz', binary=True)

    # names of all supported features
    supported_feats = ['bag_of_function_words', 'bag_of_pos_trigrams',
                       'bag_of_pos_bigrams', 'bag_of_pos_unigrams',
                       'bag_of_trigrams', 'bag_of_bigrams',
                       'bag_of_unigrams', 'topic_model_scores',
                       'characters_per_word', 'unique_words_ratio',
                       'words_per_sentence', 'word2vec_avg',
                       'word2vec_avg_max_abs', 'word2vec_max_val',
                       'liwc_scores', 'char_unigram',
                       'char_bigram', 'char_trigram']

    # features will be list of lists
    # each component list will have the same length as the list of input text
    features = []

    # for each feature, load pickle or compute values if there is no dump
    for feat in supported_feats:
        if all_features or feat in conf:
            features.extend(load_or_compute(feat))

    # transpose list of lists so its dimensions are #instances x #features
    return numpy.asarray(features).T.tolist()
