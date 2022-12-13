import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from math import log10


docIDs = []

def get_feature_names():
    for i in range(1,11):
        docIDs.append(f"doc{i}")


        
def doc2vec(text_documents):
    document_vectorization = CountVectorizer()
    document_term_matrix = document_vectorization.fit_transform(text_documents)
    return pd.DataFrame(np.transpose(document_term_matrix.toarray()),index=document_vectorization.get_feature_names(), columns=docIDs)



def w_tfLog(document_vectorization_df):
    w_tfLog_df = document_vectorization_df.copy()
    for index in document_vectorization_df.index:
        for feature in document_vectorization_df.columns:
            tf = document_vectorization_df.loc[index,feature]
            if tf != 0:
                w_tfLog_df.loc[index,feature] = tf*(1+log10(tf))
                
    return w_tfLog_df
        


get_feature_names()