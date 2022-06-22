from config.database import db
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from gensim.models.tfidfmodel import TfidfModel

from utils.cleaner import text_cleaner


class KNN():
    def __init__(self):
        self.loadFromFirestore()
        self.cleanDocument()
        self.createTFIDFModel()

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

        print(self.document_cleaned)

    def createTFIDFModel(self):
        self.dictionary = Dictionary(self.document_cleaned)
        self.num_docs = self.dictionary.num_docs
        self.num_terms = len(self.dictionary.keys())

        corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in self.document_cleaned]

        self.tfidf = TfidfModel(corpus_bow)
        corpus_tfidf = self.tfidf[corpus_bow]

        self.corpus_tfidf_sparse = corpus2csc(
            corpus_tfidf, self.num_terms, num_docs=self.num_docs).T

        print("TFIDF is created")

    def proses(self, teks, k):
        self.model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.model.fit(self.corpus_tfidf_sparse)

        test_dokumen = pd.DataFrame({"dokumen": [teks]})
        test_dokumen = test_dokumen.dokumen.dropna().reset_index(drop=True)
        test_dokumen = test_dokumen.apply(lambda x: text_cleaner(x).split())

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
