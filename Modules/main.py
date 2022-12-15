from read_docs import read10documents
from text_preprocessing import preprocessing
import tfidf_matrix
from positional_index import positional_index
from document_vectorization import doc2vec, w_tfLog
import phrase_query_results

# Read 10 documents
docs = read10documents()

# Documents Preprocessing
tokenized_docs,text_docs = preprocessing(docs)


# Documents vectorization
print("\n\n")
print("Term Frequency(TF)\n".center(75))
document_vectroization_df = doc2vec(text_docs)
print(document_vectroization_df)
print("\n")


#w tf(1+log tf)
print("\nw tf(1 + log tf)\n".center(75))
w_tfLog_df = w_tfLog(document_vectroization_df)
print(w_tfLog_df)
print("\n")


# Documents df and idf
df_idf_df = tfidf_matrix.compute_df_and_idf_manually(document_vectroization_df)
print(df_idf_df)
print("\n")

# Documents tf*idf

print("tf*idf\n".center(75))
tfidf_df = tfidf_matrix.compute_tfidf_manually(df_idf_df, document_vectroization_df)
print(tfidf_df)
print("\n")


# Documents length

print("Document Lengths\n".center(75))
docsLength_df = tfidf_matrix.compute_docsLength(tfidf_df)
print(docsLength_df)
print("\n")


# Documents Normalized tf*idf

print("Normalized tf.idf\n".center(75))
normalized_tfidf = tfidf_matrix.compute_normalized_tfidf_manually(tfidf_df, docsLength_df)
print(normalized_tfidf)
print("\n")

# Build Positional index

pos_index = positional_index(tokenized_docs)
print("\n\nPositional Index:\n")
print(pos_index)
print("\n\n")

# Get Results of the Query

op = input("Want to search for a Phrase Query? ").lower()

while op == 'y' or op == 'yes':

    phraseQuery_tokenized, phraseQuery_results_df=phrase_query_results.display_phraseQuery_df_results(df_idf_df, document_vectroization_df)
    print("\n")
    if phraseQuery_results_df.empty:
        print("No Matched Terms")
    else:
        print(phraseQuery_results_df)
    print("\n")

# Matched docs
    
    matched_docs_terms = phrase_query_results.get_matched_documents(pos_index, phraseQuery_tokenized)
    match_docs_pq = phrase_query_results.matched_phrase_query(matched_docs_terms, phraseQuery_tokenized, pos_index)
    
    print("\n")
    product_df = phrase_query_results.product_df(normalized_tfidf, phraseQuery_tokenized, phraseQuery_results_df, match_docs_pq)
    if product_df.empty:
        pass
    else:
        print(product_df)
    print("\n")
    dic = phrase_query_results.get_similiarty(product_df)
    print("\n")

    # Cosine Similiarity
    phrase_query_results.display_matched_documents(dic, match_docs_pq, text_docs)

    op = input("Want to search for a Phrase Query? ").lower()



