import string

import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)


# function to remove punctuation
def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


def remove_extra_whitespace_tabs(text):
    # pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


def to_lowercase(text):
    return text.lower()


def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()


train = pd.read_csv('DataSet/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

# 建一个dic 把所有的单词出现的次数做出来
dic_word_freq = {}

print(len(train.review))

for n in range(len(train.review)):
    # first remove html tag
    text = remove_html_tags(train.review[n])
    text = to_lowercase(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace_tabs(text)
    text = nltk.word_tokenize(text)
    filtered_words = [word for word in text if word not in stopwords.words('english')]
    for word in filtered_words:
        if word in dic_word_freq:
            dic_word_freq[word] += 1
        else:
            dic_word_freq[word] = 1

print(dic_word_freq)
result = sorted(dic_word_freq, key=dic_word_freq.get, reverse=True)
prop_result = result[:4096]
print(len(prop_result))
print(prop_result)
f = open("result.txt", "a")
f.write(str(prop_result))
f.close()