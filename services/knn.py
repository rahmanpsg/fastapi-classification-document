import json
import random
from config.database import db
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import _safe_indexing, indexable
from itertools import chain
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from gensim.models.tfidfmodel import TfidfModel
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

        # save jurnals_dict to json
        # with open('jurnals_dict.json', 'w') as f:
        #     json.dump(jurnals_dict, f)

        self.df = pd.DataFrame(jurnals_dict)

        print("total data loaded :", self.df.shape)

    def loadFromJson(self):
        print("load data from json")

        with open('jurnals_dict.json', 'r') as f:
            jurnals_dict = json.load(f)

        self.df = pd.DataFrame(jurnals_dict)

        print("total data loaded :", self.df.shape)

    def cleanDocument(self):
        self.X = pd.DataFrame([item['text']
                               for item in self.df.fileData.values], columns=['text'])

        # load X from csv
        # self.X = pd.read_csv('X.csv')

        self.document_cleaned = self.X.text.dropna().reset_index(drop=True)
        self.document_cleaned = self.document_cleaned.apply(
            lambda x: text_cleaner(x).split())

        self.y = self.df.prodi.dropna().reset_index(drop=True)
        # load y from csv
        # self.y = pd.read_csv('y.csv')

        print(self.y)

        print("document cleaned")

        # save X and y
        # self.X.to_csv('X.csv', index=False)
        # self.y.to_csv('y.csv', index=False)

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

        print(self.corpus_tfidf_sparse.shape)

        print(self.corpus_tfidf_sparse[0])

        # print(self.X.text[0])

        # text = self.X.text[0]

    # for doc in corpus_bow:
    #     print([[self.dictionary[id], freq] for id, freq in doc])

    # for doc in self.tfidf[corpus_bow]:
    #     print([[self.dictionary[id], np.around(freq, decomal=2)]
    #           for id, freq in doc])

    def proses(self, teks, k):
        self.model = NearestNeighbors(n_neighbors=k, n_jobs=-1)

        self.model.fit(X=self.corpus_tfidf_sparse, y=self.y)

        test_dokumen = pd.DataFrame({"dokumen": [teks]})
        test_dokumen = test_dokumen.dokumen.dropna().reset_index(drop=True)
        test_dokumen = test_dokumen.apply(lambda x: text_cleaner(x).split())

        print(test_dokumen)

        # test corpus from created dictionary
        test_corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in test_dokumen]

        # test tfidf values from created tfidf model
        test_corpus_tfidf = self.tfidf[test_corpus_bow]

        # for doc in test_corpus_tfidf:
        #     print([[self.dictionary[id], np.around(freq, decimals=2)]
        #           for id, freq in doc])

        # test sparse matrix
        test_corpus_tfidf_sparse = corpus2csc(
            test_corpus_tfidf, self.num_terms).T

        print(test_corpus_tfidf_sparse)

        distances, indices = self.model.kneighbors(test_corpus_tfidf_sparse)

        print(distances[0])

        df_hasil = self.df.loc[indices[0]]
        df_hasil['jarak'] = distances[0]

        hasil_tfidf = []

        for doc in self.document_cleaned.loc[indices[0]]:

            text = ' '.join(doc)

            # Memecah setiap kata
            keywords = re.findall(r'[a-zA-Z]\w+', text)

            df = pd.DataFrame(list(set(keywords)),
                              columns=['keyword'])

            # df = pd.DataFrame(test_dokumen.values[0],
            #                   columns=['keyword'])

            df['count'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[0])
            df['tf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[1])
            df['idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[2])
            df['tf_idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[3])

            # remove index where keyword not in list
            df = df.drop(
                df[df.keyword.isin(test_dokumen.values[0]) == False].index)

            # add keyword if not in list
            for keyword in test_dokumen.values[0]:
                if keyword not in df.keyword.values:
                    df = df.append(
                        {'keyword': keyword, 'count': 0, 'tf': 0, 'idf': 0, 'tf_idf': 0}, ignore_index=True)

            # print(df)

            hasil_tfidf.append(df.to_dict(orient='records'))

        # print(hasil_tfidf)
        df_hasil['hasil_tfidf'] = hasil_tfidf

        return df_hasil.to_dict(orient='records')

    def weightage(self, word, text, number_of_documents=1):
        word_list = re.findall(word, text)
        number_of_times_word_appeared = len(word_list)
        tf = number_of_times_word_appeared/float(len(text))
        idf = np.log((number_of_documents) /
                     float(number_of_times_word_appeared))
        tf_idf = tf*idf
        return number_of_times_word_appeared, tf, idf, tf_idf

    def calculateTFIDF(self):
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(self.X.text.values)
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
                              columns=['keyword'])

            df['count'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[0])
            df['tf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[1])
            df['idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[2])
            df['tf_idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[3])

            response.append(df.to_dict('records'))
            print('oke', len(response))

        return response

    def training(self):
        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.corpus_tfidf_sparse, self.y, test_size=0.20, random_state=12345)

        # random number
        r = random.randint(10000, 99999)

        cv = ShuffleSplit(random_state=r, test_size=0.20)
        arrays = indexable(self.corpus_tfidf_sparse, self.y)
        train, test = next(cv.split(X=self.corpus_tfidf_sparse))
        iterator = list(chain.from_iterable((
            _safe_indexing(a, train),
            _safe_indexing(a, test),
            train,
            test
        ) for a in arrays)
        )

        X_train, X_test, train_is, test_is, y_train, y_test, _, _ = iterator

        classifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
        cr = classification_report(y_test, y_pred, output_dict=True, )

        score = cr['accuracy']

        del cr['accuracy']

        df_hasil = self.df.loc[test_is]
        df_hasil['prediksi'] = y_pred

        response = {
            'classification_report': cr,
            'score': score,
            'data': df_hasil.to_dict(orient='records')
        }

        return response
