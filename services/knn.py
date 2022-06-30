import json
from config.database import db
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from gensim.models.tfidfmodel import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleaner import text_cleaner
import re


class KNN():
    def __init__(self):
        self.loadFromFirestore()
        self.cleanDocument()
        self.createTFIDFModel()
        # self.calculateTFIDF()

    def loadFromFirestore(self):
        print("load data from firestore")
        jurnals = list(db.collection(u'jurnals').stream())

        jurnals_dict = list(map(lambda x: x.to_dict(), jurnals))

        self.df = pd.DataFrame(jurnals_dict)

        print("total data loaded :", len(jurnals_dict))

    def cleanDocument(self):
        self.df_text = pd.DataFrame([item['text']
                                     for item in self.df.fileData.values], columns=['text'])

        self.document_cleaned = self.df_text.text.dropna().reset_index(drop=True)
        self.document_cleaned = self.document_cleaned.apply(
            lambda x: text_cleaner(x).split())

        print("document cleaned")

    def createTFIDFModel(self):
        self.dictionary = Dictionary(self.document_cleaned)
        self.num_docs = self.dictionary.num_docs
        self.num_terms = len(self.dictionary.keys())

        corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in self.document_cleaned]

        self.tfidf = TfidfModel(corpus_bow)
        self.corpus_tfidf = self.tfidf[corpus_bow]

        self.corpus_tfidf_sparse = corpus2csc(
            self.corpus_tfidf, self.num_terms, num_docs=self.num_docs).T

        print("TFIDF is created")

        print(self.tfidf.num_docs)

        # print(self.df_text.text[0])

        # text = self.df_text.text[0]

    # for doc in corpus_bow:
    #     print([[self.dictionary[id], freq] for id, freq in doc])

    # for doc in self.tfidf[corpus_bow]:
    #     print([[self.dictionary[id], np.around(freq, decomal=2)]
    #           for id, freq in doc])

    def calculateTFIDF(self):
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(self.df_text.text.values)
        # df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

        # # df_json = df.to_json(orient='records')

        # print(df.to_json(orient='records', lines=True))

        # d = {self.dictionary.get(
        #      id): value for doc in self.corpus_tfidf for id, value in doc}

        # print(json.dumps(d))

        # # json encode the dictionary
        response = []
        for doc in self.document_cleaned:
            # text = ' '.join(self.document_cleaned[0])
            text = ' '.join(doc)

            # Memecah setiap kata
            keywords = re.findall(r'[a-zA-Z]\w+', text)

            df = pd.DataFrame(list(set(keywords)),
                              columns=['keywords'])

            df['total kemunculan'] = df['keywords'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[0])
            df['tf'] = df['keywords'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[1])
            df['idf'] = df['keywords'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[2])
            df['tf_idf'] = df['keywords'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[3])

            response.append(df.to_dict('records'))
            print('oke ', len(response))

        return json.dumps(response)

    def weightage(self, word, text, number_of_documents=1):
        word_list = re.findall(word, text)
        number_of_times_word_appeared = len(word_list)
        tf = number_of_times_word_appeared/float(len(text))
        idf = np.log((number_of_documents) /
                     float(number_of_times_word_appeared))
        tf_idf = tf*idf
        return number_of_times_word_appeared, tf, idf, tf_idf

    def proses(self, teks, k):
        self.model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.model.fit(X=self.corpus_tfidf_sparse)

        test_dokumen = pd.DataFrame({"dokumen": [teks]})
        test_dokumen = test_dokumen.dokumen.dropna().reset_index(drop=True)
        test_dokumen = test_dokumen.apply(lambda x: text_cleaner(x).split())

        print(test_dokumen)

        # test corpus from created dictionary
        test_corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in test_dokumen]

        # test tfidf values from created tfidf model
        test_corpus_tfidf = self.tfidf[test_corpus_bow]

        # test sparse matrix
        test_corpus_tfidf_sparse = corpus2csc(
            test_corpus_tfidf, self.num_terms).T

        distances, indices = self.model.kneighbors(test_corpus_tfidf_sparse)

        df_hasil = self.df.loc[indices[0]]
        df_hasil['jarak'] = distances[0]

        return df_hasil.to_json(orient='records')
