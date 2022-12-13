import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log10
import math


docIDs = []
number_of_docs = 10


def get_feature_names():
    for i in range(1,11):
        docIDs.append(f"doc{i}")
        


def intilaize_df_idf_df():
    dic = {
    'df':[],
    'idf':[]
    }

    df = pd.DataFrame(dic)
    return df



def intilaize_tfidf_df():
    dic = {}
    for i in docIDs:
        dic[i] = []
    df = pd.DataFrame(dic)
    return df


def intiliaize_docsLength_feature_names():
    feature_names = []
    for i in range(1,11):
        feature_names.append('d'+str(i)+' length')
    return feature_names


def get_tfidf_vectroizer(documents, norm='l2'):
    doc_tfidf = TfidfVectorizer(norm=norm)
    document_tfidf = doc_tfidf.fit_transform(documents)
    return doc_tfidf, document_tfidf



def compute_df_and_idf_sklearn(documents, document_vectroized_df):
    tfidf_vectorizer, _ = get_tfidf_vectroizer(documents)
    ind = 0
    df = intilaize_df_idf_df()
    for word in document_vectroized_df.index:
        df.loc[len(df.index)] = [document_vectroized_df.loc[word].sum(), tfidf_vectorizer.idf_[ind]]
        ind+=1
    df.index = document_vectroized_df.index
    return df


def compute_tfidf_sklearn(documents):
    tfidf_vectorizer, document_tfidf = get_tfidf_vectroizer(documents)
    df = pd.DataFrame(np.transpose(document_tfidf.toarray()), columns=docIDs)
    df.index = tfidf_vectorizer.get_feature_names()
    return df



def compute_normalize_tfidf_sklearn(documents):
    vectorizer, document_tfidf = get_tfidf_vectroizer(documents, 'l1')
    df = pd.DataFrame(np.transpose(document_tfidf.toarray()), columns=docIDs)
    df.index = vectorizer.get_feature_names()
    return df
    

def compute_df_and_idf_manually(document_vectroized_df):
    df = intilaize_df_idf_df()
    for word in document_vectroized_df.index:
        tf = document_vectroized_df.loc[word].sum()
        df.loc[len(df.index)] = [document_vectroized_df.loc[word].sum(), log10(number_of_docs/tf)]
    df.index = document_vectroized_df.index
    return df


def compute_tfidf_manually(dfidf_df, document_vectroized_df):
    tfidf_df = intilaize_tfidf_df()
    for word in dfidf_df.index:
        tfidf_df.loc[len(tfidf_df)] =  (dfidf_df.loc[word]['idf'] * document_vectroized_df.loc[word]).values
    tfidf_df.index = document_vectroized_df.index
    return tfidf_df 


def compute_docsLength(tfidf_df):
    feature_names = intiliaize_docsLength_feature_names()
    doc_lengths = []
    for column in tfidf_df.columns:
        arr = tfidf_df[column].values**2
        compute_docLen = math.sqrt(np.sum(arr))
        doc_lengths.append(compute_docLen)
    docLen_df = pd.DataFrame(doc_lengths, index=feature_names)
    return docLen_df


def compute_normalized_tfidf_manually(tfidf_df, docLen_df):
    fet_name = intiliaize_docsLength_feature_names()
    counter = 0
    normalized_tfidf_vals = []
    normalized_tfidf_df = intilaize_tfidf_df()
    for column in tfidf_df.columns:
        length = len(tfidf_df[column])
        for index in range(length):
            normalized_tfidf_vals.append(tfidf_df[column][index] / docLen_df.loc[fet_name[counter]].values[0])
        normalized_tfidf_df[docIDs[counter]] = normalized_tfidf_vals 
        counter+=1
        normalized_tfidf_vals.clear()
    normalized_tfidf_df.index = tfidf_df.index
    return normalized_tfidf_df

get_feature_names()
    
    