'''
File: sentiment_analysis.py
Author: Vishaal Yalamanchali
Purpose: Created to test pickling and importing serialized machine learning models.
'''

from tokenization import tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from reduce_header_df import reduce_mem_usage
import settings
from evaluation import evaluate

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def decode_sentimentC(label):
    return label.lower()

def predict(vectoriser, model, text):
    # Predict the sentiment
    listD = []
    listD = tokenize(str(text).lower())
    textdata = vectoriser.transform(listD)
    sentiment = model.predict(textdata)
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1,2], ["Negative","Neutral","Positive"])
    print(df.sentiment)
    return df
# decode_map = {0: "negative", 2: "neutral", 4: "positive"}
# def decode_sentiment(label):
#     return decode_map[int(label)]

# cols = ["sentiment", "text"]
# encoding = 'latin'


# n = 5
# num_lines = 1600000
# skip_idx = [x for x in range(1, num_lines) if x % n != 0]


# TRAIN_SIZE = 0.8    

# ### --- DATA PREPROCESSING
# df = pd.read_csv('/Users/vishaalyalamanchali/Desktop/twitter-sentiment-analysis/data/training.1600000.processed.noemoticon.csv',encoding=encoding,names=cols, 
# nrows=320000, usecols=[0,5],skiprows=skip_idx)
# nf = pd.read_csv('/Users/vishaalyalamanchali/Desktop/twitter-sentiment-analysis/data/Tweets.csv',encoding=encoding,names=cols, usecols=[1,10])
# # print(len(nf))
# # print(len(df))
# # print(nf.head(100))
# df.sentiment = df.sentiment.apply(lambda x: decode_sentiment(x))
# frames = [df,nf]
# df = pd.concat(frames)
# # print(len(df))

# df.text = df.text.apply(lambda x: tokenize(x))
# df, NAlist = reduce_mem_usage(df)
# df = df.sample(frac=1).reset_index(drop=True)

# print(NAlist)
# print(df.head(10000))
# print()


def main():

    # train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE,
    #                                      random_state=42) # Splits Dataset into Training and Testing set
    # print("Train Data size:", len(train_data))
    # print("Test Data size", len(test_data)) 
    # # documents = [_text.split() for _text in df_train.text] 
    # train_data.head(10)

    # documents = [_text.split() for _text in train_data.text] 
    # w2v_model = gensim.models.word2vec.Word2Vec(size=settings.W2V_SIZE, 
    #                                         window=settings.W2V_WINDOW, 
    #                                         min_count=settings.W2V_MIN_COUNT, 
    #                                         workers=8)
    # w2v_model.build_vocab(documents)
    # words = w2v_model.wv.vocab.keys()
    # vocab_size = len(words)
    # print("Vocab size", vocab_size)

    # w2v_model.train(documents, total_examples=len(documents), epochs=settings.W2V_EPOCH)

    # # print(w2v_model.wv.most_similar("no"))

    # labels = train_data.sentiment.unique().tolist()
    # labels.append(settings.NEUTRAL)
    # print(labels)

    # encoder = LabelEncoder()
    # encoder.fit(train_data.sentiment.tolist())

    # X_train = train_data.text
    # X_test = test_data.text

    # y_train = encoder.transform(train_data.sentiment.tolist())
    # y_test = encoder.transform(test_data.sentiment.tolist())

    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)

    # vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    # vectoriser.fit(X_train)
    # print(f'Vectoriser fitted.')    
    # print('No. of feature_words: ', len(vectoriser.get_feature_names()))

    # X_train = vectoriser.transform(X_train)
    # X_test  = vectoriser.transform(X_test)

    # LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    # LRmodel.fit(X_train, y_train)
    # evaluate(LRmodel,X_test,y_test)

    # file = open('vector-gram.pickle','wb')
    # pickle.dump(vectoriser, file)
    # file.close()

    # file = open('linear-regression.pickle','wb')
    # pickle.dump(LRmodel, file)
    # file.close()
    
    file = open('LR.pkl', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    # print(LRmodel, type(LRmodel))
    
    file = open('vectoriser.pkl','rb')
    vectoriser = pickle.load(file)
    file.close()

    # print(vectoriser, type(LRmodel))

    text = []
    inputT = ""
    while(inputT != "0"):
        inputT = input("enter text that you want to evaluate: ")
        text.append(inputT)
        dfN = predict(vectoriser, LRmodel, text)
        #dfT = predict(vectoriser, RFmodel, text)
        print(dfN.head())
        text.clear()
    

if __name__ == "__main__":
    main()