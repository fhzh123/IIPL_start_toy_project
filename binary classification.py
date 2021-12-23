import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk import word_tokenize
nltk.download('punkt')

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator


# ==================== step1 : load data ====================

# train dataset
train = pd.read_csv(r"C:\Users\cocor\VScode_workplace\iiplab\start_toy_project\dataset\train.csv")
# test dataset
test = pd.read_csv(r"C:\Users\cocor\VScode_workplace\iiplab\start_toy_project\dataset\test.csv")


# ==================== step2 : data preprocessing ====================
# 중복제거
train = train.drop_duplicates()
test = test.drop_duplicates()

# 알파벳과 공백 제외, 제거
train['comment'] = [' '.join(re.sub('[^a-zA-Z .\']', '', word) for word in row.split()) for row in train['comment']]
test['comment'] = [' '.join(re.sub('[^a-zA-Z ]', '', word) for word in row.split()) for row in test['comment']]

# stopword 제거 (stopword_ nltk stopwords 중 일부 사용)
stopwords=['i', "i'm", "i've", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'once', 'there', 'when', 'where', 'why', 'how']
train['comment'] = [' '.join(word for word in row.split() if word.lower() not in stopwords) for row in train['comment']]
test['comment'] = [' '.join(word for word in row.split() if word.lower() not in stopwords) for row in test['comment']]

# positive -> 1, negative -> 0
train['label']=0
train.loc[train.sentiment=='positive','label']=1
train.loc[train.sentiment=='negative','label']=0
test['label']=0
test.loc[test.sentiment=='positive','label']=1
test.loc[test.sentiment=='negative','label']=0

train[['comment','label']].to_csv('train_data.csv', index=False)
test[['comment','label']].to_csv('test_data.csv', index=False)


# ==================== step3 : tokenizing & padding ====================
#TEXT = data.Field(sequential=True, use_vocab=True, tokenize=str.split, lower=True, batch_first=True)
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=word_tokenize, lower=True, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=False, is_target=True)

trainset, testset = TabularDataset.splits(path='.', train='train_data.csv', test='test_data.csv',
                                              format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

TEXT.build_vocab(trainset, min_freq=3) #최소 2회 등장
text_size = len(TEXT.vocab) #단어집합 크기
#print(TEXT.vocab.stoi)

BATCH_SIZE=64
trainset, valset = train_data.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset),
                                                             batch_size=BATCH_SIZE, shuffle=True, repeat=False)

