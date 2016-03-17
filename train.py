import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import treetaggerwrapper
import time
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from lexical_features import punctuation
from lexical_features import vocabulary
from lexical_features import phrases

#convert text to tags using TreeTagger wrapper for Python
def text_to_tags(text):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/yassine/EMSE 2015-2016/Projet Recherche/tree-tagger-linux-3.2')
    tags = treetaggerwrapper.make_tags(tagger.tag_text(unicode(text,encoding='utf-8')))
    pos_tags = []
    for pos in tags:
        pos_tags.append(pos[1])
    return " ".join(pos_tags)


path = '/home/yassine/EMSE 2015-2016/Projet Recherche/Author-Verification-/corpus-english-sample'
filenames = []
filenames_tags = []
filenames_first = []
filenames_last = []

start_time = time.time()
for root, dirs, files in os.walk(path):
    for directory in sorted(dirs):
        #print directory[-2:]  #corpus
        for r, d, f in os.walk(path+'/'+directory):
            
            print 'Processing problem: '+directory
            
            #get documents for the problem: e.g known01.txt , known02.txt ... and unknown.txt
            for name in sorted(f):
                if  not "tags" in name and not "first" in name and not "last" in name:
                    filenames.append(path+'/'+directory+'/'+name) #populating filenames array
                
                #POS Processing
                #first_tags = []
                #last_tags = []
                #text_tags = []
                
                #if not "tags" in name and not "first" in name and not "last" in name:
                    
                    #print "[Part-Of-Speech] Processing file: "+ directory +"/"+ name + "... %f seconds elapsed" % (time.time() - start_time) 
                   
                   ##converting text to pos tags sentence by sentence to keep first and last tags seperately
                    #f = open(path+'/'+directory+'/'+name,"r")
                    #train = f.read()
                    #sentences = train.replace('\n','')
                    #sentences = train.replace('.','\n').splitlines()
                    #for s in sentences:
                        #s = text_to_tags(s)
                        #first_tags.append(" ".join(s.split()[0:1])) #add first pos tags of each sentence to 'first_tags'
                        #last_tags.append(" ".join(s.split()[-1:]))  #add last pos tags of each sentence to 'last_tags'
                        #text_tags.append(s)       #add all pos tags of each sentence to 'lext_tags'
                    
                    #f_tags = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_tags.txt',"w") #write entire text in tags to file
                    #f_tags.write(" ".join(text_tags))
                    #f_tags.close()   
                    #filenames_tags.append(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_tags.txt') 
                    
                    #f_first = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_first.txt',"w") #write first tags to file
                    #f_first.write(" ".join(first_tags))
                    #f_first.close() 
                    #filenames_first.append(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_first.txt') 
                    
                    #f_last = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_last.txt',"w") #write last tags to file
                    #f_last.write(" ".join(last_tags))
                    #f_last.close() 
                    #filenames_last.append(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_last.txt') 
                    
              
        ### TASK 1 ###
        #Compute the documents representations using the paths stored in the 'filenames' array
        
            #char 8-gram tfidf feature
            print '[8-gram] calculting tfidf for all documents...'
            tfidf = TfidfVectorizer(input='filename',use_idf=True,analyzer='char',ngram_range=(8,8))
            eight_char = tfidf.fit_transform(filenames).toarray() 
            
            #Lexical features: create 3 features vector: punct, vocab, phrase
            print '[Lexical Features] Calculating punct/vocab/phrases stats for all documents...'
            i = 0
            for doc in filenames:
                    f = open(doc,"r")
                    train = f.read()
                    tmp1 = punctuation(train)
                    tmp2 = vocabulary(train)
                    tmp3 = phrases(train)
                    if i == 0: 
                            punct = tmp1
                            vocab = tmp2
                            phrase = tmp3
                            i += 1
                    else: 
                            punct = np.concatenate((punct,tmp1))
                            vocab = np.concatenate((vocab,tmp2))
                            phrase = np.concatenate((phrase,tmp3))

        ### TASK 2 ###
        #compute the distance between known and unknown documents for each feature
            
            #distance between known and unknown doc on 8-char feature
            print 'calculating distances for 8-gram feature...'
            eight_charM = cosine_similarity(eight_char)
            
            print 'calculating distances for lexical features...'
            punctM = cosine_similarity(punct)
            dist = DistanceMetric.get_metric('euclidean')
            vocabM = dist.pairwise(vocab)
            phraseM = np.corrcoef(phrase)
        
        ### TASK 3 ###
        #strore the new vector of distance as a new observation for the specific problem (in 'directory')
            
            #take the mean of the all the distances between known documents and the unknown doc
            print eight_charM[(len(eight_charM)-1),:(len(eight_charM)-1)].mean()
            print punctM[(len(punctM)-1),:(len(punctM)-1)].mean()
            print vocabM[(len(vocabM)-1),:(len(vocabM)-1)].mean()
            print phraseM[(len(phraseM)-1),:(len(phraseM)-1)].mean()

            #empty 'filenames' arrays
            filenames = []
            filenames_tags = []
            filenames_first = []
            filenames_last = []
    
