import math
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report,\
precision_recall_fscore_support, roc_auc_score
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

import string
exclude = set(string.punctuation)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
lemmatizer = WordNetLemmatizer()

from nltk.util import ngrams

#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0

idf_file_name = 'docIDFDict_ngram.pickle'
doc_length_file_name = 'avgDocLength_ngram.pickle'


def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f:
        passage = line.strip().lower().split("\t")[2]
        fw.write(passage+"\n")
    f.close()
    fw.close()

def get_wordnet_pos(pos_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
    """
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN

def lemma(tokens):
    # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
    pos_tokens = pos_tag(tokens)
    #[pos_tag(token) for token in tokens]
    # lemmatization using pos tagg   
    # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']),
    #... ie [original WORD, Lemmatized word, POS tag]
    lemma_word = [lemmatizer.lemmatize(word,get_wordnet_pos(pos_tag)) for (word,pos_tag) in pos_tokens]
    return lemma_word

# new function to process the sentence to tokenise words. This will include
    # word tokeniser
    # lower case
    # stop words removal
    # stemming / lematisation
    # punctuation removal
def tokenise_word(line,is_trigram = False,is_lemma = False):
    tokens = word_tokenize(line.lower().strip())
    #filtered_sentence = [w for w in tokens if not w in stop_words]
    filtered_words = [w for w in tokens if not w in exclude and not w in stop_words]
    if is_lemma:
        stemmed_words = lemma(filtered_words)
    else:
        stemmed_words = [stemmer.stem(w) for w in filtered_words]
    
    # bigrams    
    bi_garms = list(ngrams(stemmed_words,2))
    bi_garms = ('_'.join(w) for w in bi_garms)
    
    if is_trigram:
    # trigrams
        tri_grams = list(ngrams(stemmed_words,3))
        tri_grams = ('_'.join(w) for w in tri_grams)
    
    stemmed_words.extend(bi_garms)
    if is_trigram:
        stemmed_words.extend(tri_grams)
    return stemmed_words

# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula 
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e, min_doc = 5) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        #doc = line.strip().split(delimiter)
        doc = tokenise_word(line, False, False)
        totalDocLength += len(doc)

        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%10000==0):
            print(numOfDocuments)                

    print("Number of unique words before reduction:",len(docFrequencyDict))
    for k, v in list(docFrequencyDict.items()):
        if v < min_doc:
            del docFrequencyDict[k]
    
    print("Number of unique words after reduction:",len(docFrequencyDict))        
    
    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] +0.5) / (docFrequencyDict[word] + 0.5), base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments


    
    pickle_out = open(idf_file_name,"wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()

    with open(doc_length_file_name, "wb") as avgDocLength_file:
        pickle.dump(avgDocLength, avgDocLength_file)
    
    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)



#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength

    #query_words= Query.strip().lower().split(delimiter)
    #passage_words = Passage.strip().lower().split(delimiter)
    query_words= tokenise_word(Query, False, False)
    passage_words = tokenise_word(Passage, False, False)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%10000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()
    
#The following line reads each line from validate file and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnValidationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    tempsinglequery = [] # this will store all lines of single query as list of lists
    temp_predicted_binary_score = [] # temp list for question wise predicted binary scores
    actual_binary_score = []
    predicted_binary_score = []
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        # add absolute score for this line
        tokens.append(str(score))
        # add place holder for predicted binary score
        tokens.append('0')
        # single list for actual binary scores
        actual_binary_score.append(int(tokens[3]))
        # place holder for question wise binary scores
        temp_predicted_binary_score.append(0)
        # add the whole line to the list which can be written later
        tempsinglequery.append(tokens)
        tempscores.append(score)
        lno += 1
        if(lno % 10 == 0):
            max_score_pos = np.argmax(tempscores)
            tempsinglequery[max_score_pos][(len(tokens) - 1)] = '1'
            temp_predicted_binary_score[max_score_pos] = 1
            predicted_binary_score.extend(temp_predicted_binary_score)
            #print(tempsinglequery)
            fw.writelines('\t'.join(line) + '\n' for line in tempsinglequery)
            # instanciate for next query
            tempsinglequery = []
            tempscores=[]
            temp_predicted_binary_score = []
        if(lno % 10000 == 0):
            print(lno)
    print(lno)
    f.close()
    fw.close()
    return actual_binary_score, predicted_binary_score


if __name__ == '__main__' :

    inputFileName = "D:/Data Science/MS_AI_Challenge/interim/traindata.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    validationFileName = "D:/Data Science/MS_AI_Challenge/interim/validationdata.tsv"
    testFileName = "D:/Data Science/MS_AI_Challenge/data/eval1_unlabelled.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
    corpusFileName = "D:/Data Science/MS_AI_Challenge/interim/corpus.tsv" 
    valOutputFile = "D:/Data Science/MS_AI_Challenge/interim/val_output.tsv"
    outputFileName = "D:/Data Science/MS_AI_Challenge/output/answer.tsv"
    
    with open(idf_file_name, "rb") as idf_file:
        docIDFDict = pickle.load(idf_file)
        
    with open(doc_length_file_name, "rb") as avgDocLength_file:
        avgDocLength = pickle.load(avgDocLength_file)
    #GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
# =============================================================================
#     print("Corpus File is created.")
#     IDF_Generator(corpusFileName)   # Calculates IDF scores. 
#     #RunBM25OnTestData(testFileName,outputFileName)
#     print("IDF Dictionary Generated.")
# =============================================================================

    
    actual_binary_score, predicted_binary_score = RunBM25OnValidationSet(validationFileName,valOutputFile)
    print("Validation output file created. ")
    
    print(confusion_matrix(actual_binary_score,predicted_binary_score))
    print(classification_report(actual_binary_score,predicted_binary_score))
    print(roc_auc_score(actual_binary_score,predicted_binary_score))
    print(precision_recall_fscore_support(actual_binary_score,predicted_binary_score,\
                                          pos_label = 1, average = 'binary'))
    
    
    #fpr, tpr, thresholds = roc_curve(actual_binary_score,predicted_binary_score, pos_label=2)
       
    #RunBM25OnEvaluationSet(testFileName,outputFileName)
    #print("Submission file created. ")

