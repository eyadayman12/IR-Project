from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import contractions
import re


stop_words = stopwords.words('english')

'''
A function to lowercase all the characters in a document
'''

def lowercase_document(list_of_documents):
    lowercase_documents = []
    for document in list_of_documents:
        lowercase_documents.append(document.lower())
    return lowercase_documents


'''
A function to apply contraction
contraction is to change the word to shortened version
for example word aren't applying contraction will changed it to are not and so on
this would be useful when removing punctations so it will not remove the comma which will lead unexist word
'''

def apply_contraction(list_of_documents):
    documents_list = []
    for document in list_of_documents:
        document_after_applying_contraction = ""
        for word in document.split():
            document_after_applying_contraction=document_after_applying_contraction + contractions.fix(word) + ' '
        documents_list.append(document_after_applying_contraction)
    return documents_list


'''
A function to remove all punctations
'''
def remove_punctuations(list_of_documents):
    document_list = []
    for document in list_of_documents:
        cleaned_data = document
        cleaned_data = re.sub(r'[^\w\s.,]', '', cleaned_data)
        document_list.append(cleaned_data)
    return document_list


'''
A function to tokenize the document
'''
def tokenize(list_of_documents):
    document_list = []
    for document in list_of_documents:
        document_list.append(word_tokenize(document))
    return document_list


'''
A function to remove all the stopwords except in and to
'''
def remove_stopwords(tokenized_documents):
    documents_tokinzed_list = []
    text_documents = []
    for doc in tokenized_documents:
        filtered_sentence = []
        cleaned_sentence = ""
        for word in doc:
            if word not in stop_words or word == "in" or word == "to" or word == "where":
                cleaned_sentence = cleaned_sentence + word + ' '
                filtered_sentence.append(word)
        documents_tokinzed_list.append(filtered_sentence)
        text_documents.append(cleaned_sentence)
    return documents_tokinzed_list, text_documents



def preprocessing(docs):
    docs = lowercase_document(docs)
    docs = apply_contraction(docs)
    docs = remove_punctuations(docs)
    docs = tokenize(docs)
    tokenized_documents, text_documents = remove_stopwords(docs)
    return tokenized_documents, text_documents
