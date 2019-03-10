import os
import sys
import torch
import json
import csv
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torchtext.data import TabularDataset
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import random
from sklearn.manifold import TSNE

VAL_RATIO = 0.4
TEST_RATIO = 0.25
NLP = spacy.load('en')
MAX_CHARS = 20000
stop = stopwords.words('english')

def tokenize_sent(text):
    tokenized =  sent_tokenize(text)
    return tokenized

def tokenize_word(text):
    tokenized =  word_tokenize(text)
    return tokenized
		
def shuffle_tokenized(text):
    random.shuffle(text)
    newl=list(text)
    return newl

def augmentation(data):
    augmented = []
    aug_df = pd.DataFrame({'summary':[], 'rating':[]})
    reps=[]
    for index, game in data.iterrows():
        genre = game['rating']
        
        if(genre == 'Dislike'):
            s = 38
            tok = tokenize_word(game['summary'])
        elif(genre == 'Acclaim'):
            s = 14
            tok = tokenize_word(game['summary'])
        else:
            s = 3
            tok = tokenize_sent(game['summary'])

        # if(genre in ('Turn-based strategy (TBS)')):
        #     s = 120
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Tactical')):
        #     s = 110
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Quiz/Trivia')):
        #     s = 100
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Music')):
        #     s = 45
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Arcade')):
        #     s = 17
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ("Hack and slash/Beat 'em up")):
        #     s = 21
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Racing')):
        #     s = 20
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Sport')):
        #     s = 23
        #     tok = tokenize_word(game['summary'])
        # elif(genre in ('Indie')):
        #     s = 18
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Real Time Strategy (RTS)')):
        #     s = 12
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Strategy')):
        #     s = 1
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Fighting')):
        #     s = 7
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Point-and-click')):
        #     s = 6
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Simulator')):
        #     s = 7
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Adventure')):
        #     s = 3
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Platform')):
        #     s = 2
        #     tok = tokenize_sent(game['summary'])
        # elif(genre in ('Puzzle')):
        #     s = 4
        #     tok = tokenize_sent(game['summary'])
        # else:
        #     s = 0
        #     tok = tokenize_sent(game['summary'])

        shuffled = [tok]

        #print(ng_rev)
        for i in range(s):
        #generate 11 new reviews
            shuffled.append(shuffle_tokenized(shuffled[-1]))
        for k in shuffled:
            '''create new review by joining the shuffled sentences'''
            s = ' '
            new_game = s.join(k)
            if new_game not in augmented:
                augmented.append(new_game)
                aug_df = aug_df.append(pd.DataFrame({'summary': [new_game] , 'rating': [genre]}))
            else:
                reps.append(new_game)
    return data.append(aug_df)

def convert_csv():
    with open('dataset/datasetlstm.json') as datalstm:  
        games = json.load(datalstm)

    new_csv = open('databi.csv', 'w+')
    csvwriter = csv.writer(new_csv)

    count = 0

    for g in games:
        if count == 0:
                header = g.keys()
                csvwriter.writerow(header)
                count += 1
        csvwriter.writerow(g.values())
    new_csv.close()

def text_filter(data):
    # Remove quebra de linha
    data["summary"] = \
        data.summary.str.replace("\n", " ")

    # Transforma em lowercase
    data['summary'] = data['summary'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

    # # Remove pontuacao
    data['summary'] = data['summary'].str.replace('[^\w\s]','')

    # Remove stop words
    data['summary'] = data['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Remove palavras mais frequentes
    freq = pd.Series(' '.join(data['summary']).split()).value_counts()[:10]
    freq = list(freq.index)
    data['summary'] = data['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # Remove palavras menos frequentes
    freq = pd.Series(' '.join(data['summary']).split()).value_counts()[-10:]
    freq = list(freq.index)
    data['summary'] = data['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return data

def prepare_csv(seed=999):
    df_train = pd.read_csv("dataset/dataset.csv")
    keep_col = ['summary', 'storyline', 'rating']
    keep_data = ['summary', 'rating']
    df_train = df_train[keep_col]

    df_train['summary'] = df_train['summary'].map(str) +  df_train['storyline'].map(str)
    df_train = df_train[keep_data]

    #new_f.to_csv("bi_data.csv", index=False)
    
    #new_f.dropna(inplace=True)
    #print(df_train)

    print('Before Augumentation')
    print(df_train.groupby(['rating'])['summary'].count())

    # print(df_train.columns)
    # print(df_train.head(2))

    # print(df_train['rating'].unique())

    df_train = augmentation(df_train)
    # plots(df_train)

    df_train = text_filter(df_train)

    print('After Augumentation')
    print(df_train.groupby(['rating'])['summary'].count())

    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)

    val_size = int(len(idx) * VAL_RATIO)

    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/dataset_train.csv", index=False)

    df_val = df_train.iloc[idx[:val_size], :]

    idx = np.arange(df_val.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)

    test_size = int(len(idx) * TEST_RATIO)
    df_val.iloc[idx[test_size:], :].to_csv(
        "cache/dataset_val.csv", index=False)

    df_val.iloc[idx[:test_size], :].to_csv(
        "cache/dataset_test.csv", index=False)

def plots(data):
    rating = data.groupby(by=['rating'])
    print(rating.count())

    count_series = rating.count().iloc[:,0]
    features_of_interest = pd.DataFrame({'summary': count_series})

    names = data['rating'].unique()
    values = rating.count()

    fig, axs = plt.subplots()
    axs.barh(values.axes[0], values['summary'])
    fig.suptitle('Categorical Plotting')

    plt.show()

    n = 50

    # Makes the dots colorful
    colors = np.random.rand(n)

    # Plots best-fit line via polyfit
    plt.plot(np.unique(values.axes[0]), np.poly1d(np.polyfit(values.axes[0], values['summary'], 1))(np.unique(values.axes[0])))

    # Plots the random x and y data points we created
    # Interestingly, alpha makes it more aesthetically pleasing
    plt.scatter(values.axes[0], values['summary'], c=colors, alpha=0.5)
    plt.show()

def tsne_plot(model):
    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(model.vectors)
    labels = model.itos

    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
    plt.show()

def load_dataset(test_sen=None):
    prepare_csv()
    
    # tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()

    datafields = [("summary", TEXT),
                  ("rating", LABEL)]

    # datafields = [("summary", TEXT),
    #               ('storyline', None),
    #               ("rating", LABEL)]

    train_data, valid_data, test_data = TabularDataset.splits(
                path="cache",
                train='dataset_train.csv', 
                validation='dataset_val.csv',
                test='dataset_test.csv',
                format='csv',
                skip_header=True,
                fields=datafields)

    TEXT.build_vocab(train_data, vectors="glove.6B.300d")
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    #tsne_plot(TEXT.vocab)
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    # train_data, valid_data = train_data.split()
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.summary), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)

    return TEXT, LABEL, label_size, vocab_size, word_embeddings, train_iter, valid_iter, test_iter

#prepare_csv()
