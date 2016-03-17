from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#punctiation frequence of (',' ':' ';' '(' ')' '!') in text (normaized by thenumber of sentences)
def punctuation(text):
    sentences = text.replace('\n','')
    sentences = sentences.replace('.','\n').splitlines()
    matrix = np.zeros((1, 6))
    matrix[0, 0] = text.count(',')/float(len(sentences))
    matrix[0, 1] = text.count(':')/float(len(sentences))
    matrix[0, 2] = text.count(';')/float(len(sentences))
    matrix[0, 3] = text.count('(')/float(len(sentences))
    matrix[0, 4] = text.count(')')/float(len(sentences))
    matrix[0, 5] = text.count('!')/float(len(sentences))
    return matrix

#words number of stop words in text / number of total words in text
def vocabulary(text):
    count = CountVectorizer(analyzer='word',ngram_range=(1,1),stop_words='english')
    countTotal = CountVectorizer(analyzer='word',ngram_range=(1,1))
    counter = count.fit_transform([text]).toarray()
    countT = countTotal.fit_transform([text]).toarray()
    matrix = np.zeros((1, 1))
    matrix[0, 0] = (countT.sum()-counter.sum())/float(countT.sum())

    return matrix

#Average and standard deviation of words per sentence
def phrases(text):
    sentences = text.replace('\n','')
    sentences = sentences.replace('.','\n').splitlines()
    wps = np.array([len(s.split()) for s in sentences]) #wps contains the number of words in each sentence
    matrix = np.zeros((1, 5))
    matrix[0, 0] = wps.std() #deviation
    matrix[0, 1] = wps.mean() #mean
    matrix[0, 2] = np.amin(wps) #min
    matrix[0, 3] = np.amax(wps) #max
    matrix[0, 4] = len(sentences) #number of sentences
    return matrix
