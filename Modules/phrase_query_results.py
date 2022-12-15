import pandas as pd
from text_preprocessing import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from math import log10
import sys

flag = True

def input_phraseQuery():
    phraseQuery = input("Enter Phrase Query: ")
    return phraseQuery



def initilaize_phraseQuery_results_df(): 
    phraseQuery_results_df = pd.DataFrame({
        'tf-raw':[],
        'w tf(1+logtf)':[],
        'idf':[],
        'tf*idf':[]
    })
    return phraseQuery_results_df



def display_phraseQuery_df_results(df_idf_df, document_vectroization_df):
    
    global flag
    phraseQuery = input_phraseQuery()
    phraseQuery = [phraseQuery]
    phraseQuery_tokenized, phraseQuery_text = preprocessing(phraseQuery)
    phraseQuery_results_df = initilaize_phraseQuery_results_df()
    length_of_phaseQuery = len(phraseQuery_tokenized[0])
    document_vectorization = CountVectorizer()
    document_term_matrix = document_vectorization.fit_transform(phraseQuery)
    tf_idf_list = []
    tf_idf_squared = []
    try:
        for i in range(length_of_phaseQuery):
            phraseQuery_results = []
            tf = document_term_matrix.toarray()[0][i]
            w_tf = 0
            if tf != 0:
                w_tf = 1 + log10(tf)
            idf_of_phraseQuery= df_idf_df.loc[phraseQuery_tokenized[0][i]]['idf']
            tf_idf = tf * idf_of_phraseQuery
            tf_idf_list.append(tf_idf)
            tf_idf_squared.append(tf_idf**2)
            phraseQuery_results.append(tf)
            phraseQuery_results.append(w_tf)
            phraseQuery_results.append(idf_of_phraseQuery)
            phraseQuery_results.append(tf_idf)
            phraseQuery_results_df.loc[phraseQuery_tokenized[0][i]] = phraseQuery_results
            
            
    except:
        print("a7a")
        pass
    
    
    doclen = np.sqrt(sum(tf_idf_squared))
    normalized_tfidf = [i/doclen for i in tf_idf_list]
    phraseQuery_results_df['normalized'] = normalized_tfidf
     
    try:
        for i in phraseQuery_tokenized[0]:
            document_vectroization_df.loc[i]
    except:
        flag = False
    
    return phraseQuery_tokenized,phraseQuery_results_df

        
        
def get_documents_for_each_term(pos_index, phraseQuery_tokenized):
    length_of_phaseQuery = len(phraseQuery_tokenized[0])
    doc_ids = []
    try:
        for i in range(0, length_of_phaseQuery):
            sample_pos_idx = pos_index[phraseQuery_tokenized[0][i]]
            docIDs_docPos = sample_pos_idx[1]
            docID = []
            for docIDs, docPos in docIDs_docPos.items():
                docID.append(docIDs+1)
            doc_ids.append(docID)
    except:
        pass
    
    return doc_ids


def get_matched_documents(pos_index, phraseQuery_tokenized):
    
    matched_docs = []
    if flag:
        try:
            doc_ids = get_documents_for_each_term(pos_index, phraseQuery_tokenized)
            matched_docs = set(doc_ids[0])

            for i in range(1,len(doc_ids)):
                matched_docs = matched_docs & set(doc_ids[i])
            matched_docs = list(matched_docs)
        except:
            pass
        
    return matched_docs



def matched_phrase_query(matched_docs, phraseQuery_tokenized, pos_index):

    match_docs = []
    
    try:
        if len(matched_docs) != 0:
            for doc in matched_docs:
                flag2 = True
                prev = -1
                for docID in phraseQuery_tokenized[0]:
                    current = pos_index[docID][1][doc-1][0]
                    if prev == -1:
                        prev = current
                        continue
                    else:
                        if current == prev+1:
                            prev = current
                            continue
                        else:
                            flag2 = False
                            break
                if flag2:
                    match_docs.append(doc)
                    
    except:
        sys.exit(1)
         
    return match_docs


def product_df(normalized_tfidf, phraseQuery_tokenized, phraseQuery_results_df, match_docs):
    df = pd.DataFrame()
    
    if len(match_docs) == 0:
        print("Can't generate product DF because there is no matched documents")
    
    else:
        df = pd.DataFrame()
        for i in match_docs:
            df["doc"+str(i)] = []


        feature_names = df.columns
        for word in phraseQuery_tokenized[0]:
            product = []
            for feature in feature_names:
                product.append(normalized_tfidf.loc[word].loc[feature] * phraseQuery_results_df.loc[word]['normalized'])

            df.loc[len(df.index)] = product

        add = []

        for feature in feature_names:
            add.append(df[feature].sum())

        df.loc[len(df.index)] = add

        indcies = phraseQuery_tokenized[0]
        indcies.append("Sum")
        df.index = indcies
    return df
        
       

        
        
def get_phraseQuery(phraseQuery_tokenized):
    phraseQuery = ''
    for term in phraseQuery_tokenized[0]:
        phraseQuery += term + ' '
    phraseQuery = phraseQuery[:len(phraseQuery)-1]
    return phraseQuery


def get_combined_docs_and_query(phraseQuery_tokenized, matched_docs, text_docs):
    phraseQuery = get_phraseQuery(phraseQuery_tokenized)
    returned_docs = []
    returned_docs.append(phraseQuery)
    for i in matched_docs:
        returned_docs.append(text_docs[i-1])
    return returned_docs


def get_cosine_similarity_between_query_and_docs(phraseQuery_tokenized, matched_docs, text_docs):
    
    returned_docs = get_combined_docs_and_query(phraseQuery_tokenized, matched_docs, text_docs)
    dictionary = {}
    for i in range(1, len(returned_docs)):
        corpus = []
        corpus.append(returned_docs[0])
        corpus.append(returned_docs[i])
        cv = CountVectorizer(stop_words="english") 
        X = cv.fit_transform(corpus).toarray()
        cosine = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 
        dictionary[f'cosine similarity (q , doc{str(i)})'] = cosine(X[0], X[1])
    dictionary = dict(sorted(dictionary.items(), key = lambda x: x[1], reverse = True))
    for key, value in dictionary.items():
        print(f"{key}: {value}")
        
        
def get_similiarty(df):
    dic = {}
    for column in df.columns:
        dic[column] = df.loc['Sum'][column]
    for key,value in dic.items():
        print(f"cosine similarity of (q,{key}): {value}")
    dic = dict(sorted(dic.items(), key = lambda x: x[1], reverse = True))
    return dic


def display_matched_documents(dic, match_docs, text_docs):
    if len(match_docs) != 0:
        print("returned documents: ", end='')
        for i,j in dic.items():
            print(f"{i}", end=", ")
        print("\n\n")
        for i in match_docs:
            print(f"doc{i}: {text_docs[i-1]}")
    else:
        print("No Matched documents founded from your phrase Query")