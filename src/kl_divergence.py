import re
import math
from nltk import PorterStemmer
from nltk.corpus import stopwords as stop_word
from nltk.tokenize import word_tokenize as wt


stopWord = set(stop_word.words("english"))
dots = ['.', ',', '!', '', "+", "#", "(", ")", ":", "'s", "'", '"']
slangs = {"n't": "not", "r": "are", "u": "you"}


def clean(doc, word_count={}):
    """
    Clean the documents and build the word frequency dictionary
    :param doc: string (para or sentences)
    :param word_count: it count the the occurance of each word
    :return: tokenized and Stemmed word
    """
    doc = doc.lower()
    tokens = wt(doc)

    filterWord = []
    for w in tokens:
        if w not in dots and w not in stopWord:
            if w in slangs:
                w = slangs[w]
            filterWord.append(w)

    sents = " ".join(filterWord)
    filterWord = re.findall('\w+', sents)

    ps = PorterStemmer()

    for w in filterWord:
        fword = ps.stem(w)

        word_count[fword] = word_count.get(fword, 1.0)
        word_count[fword] += 1

    return word_count


def get_distribution(doc):
    """
    Calculate the distribution of a document i.e the relative frequency of all words inside it.
    :param doc: any string document
    :return: reletive frequeny of words;
    """
    word_count = {}
    word_count = clean(doc, word_count)
    factor = 1.0 / sum(word_count.values())
    dist = {k: v * factor for k, v in word_count.items()}
    return dist


def get_distribution_corpus(documents, min_count):
    """
    Calculate the distribution of the entire corpus i.e the relative frequency of all words inside it.
    :param documents: the list of documents in the corpus
    :param min_count: minimum count to remove less occurring word
    :return:
    """
    word_count = {}
    word_count = clean(documents, word_count)

    corpus_dist = {}
    use_count = [word_count[w] for w in word_count if word_count[w] > min_count]
    total = sum(use_count)

    for key in word_count:
        if word_count[key] > min_count:
            corpus_dist[key] = word_count[key] / total

    return corpus_dist


def get_Kl_divergence(model1, model2, collection, lam, missing_val = 0.0001):
    """
    Find the KL Divergence between two distributions
    :param model1: first distribution dictionary
    :param model2: 2nd distribution dictionary
    :param collection: collection distribution dict
    :param lam: smoothing parameters
    :param missing_val: if this are not in collection then the default prob
    :return: a positive integer representing the KL divergence, higher the value less similar they are.
    """
    smoot_m2 = {key: (1-lam)*model2.get(key, 0) + lam*collection.get(key, missing_val) for key in model1}

    divergence = sum([model1[key]*math.log(model1[key]/smoot_m2[key]) for key in model1])
    return divergence


# 1. Build the list of documents to form the corpus or load from a file.
documents = ["", "", "", ""]

# 2. List of candidate sentences from the corpus whose relative importance is to be known.
sentences = ["", ""]

# 3. define the model parameters
lam = 0.3   # smoothness parameter
min_count = 3   # min word frequency to discard

# 4. Get the distribution of sentences in the corpus.
sent_distributions = [get_distribution(sent) for sent in sentences]

# 5. Get the distribution of the entire corpus
corpus_dist = get_distribution_corpus(documents, min_count)

# 6. Find the relative importance of each sentence using KL-divergence
relevance_scores = []
for sentence, sent_dist in zip(sentences, sent_distributions):
    kl_divergence = get_Kl_divergence(sent_dist, corpus_dist, corpus_dist, lam)
    relevance_scores.append(kl_divergence)

print(relevance_scores)
