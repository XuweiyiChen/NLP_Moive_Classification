import ast
import re
import string
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


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


def reformat(samples, labels):
    # build the model
    file = open("drive/MyDrive/researchProject/researchProject/result.txt")

    line = file.read().replace("\n", " ")
    file.close()
    model_vocabulary = ast.literal_eval(line)
    new = []

    for sample in samples:
        text = remove_numbers(sample)
        text = remove_html_tags(text)
        text = to_lowercase(text)
        text = remove_punctuation(text)
        text = remove_extra_whitespace_tabs(text)
        text = nltk.word_tokenize(text)
        filtered_words = [word for word in text if word not in stopwords.words('english')]
        result_list = np.zeros(4096)

        for word in filtered_words:
            if word in model_vocabulary:
                num = model_vocabulary.index(word)
                result_list[num] = 1

        result = np.resize(result_list, (64, 64))
        new.append(result)

    # labels 变成 one-hot encoding
    one_hot_labels = []

    for num in labels:
        one_hot = [0.0] * 2
        if num == 0:
            one_hot[0] = 1.0
        else:
            one_hot[1] = 1.0
        one_hot_labels.append(one_hot)
    new_labels = np.array(one_hot_labels)
    print(new)
    print(new_labels)
    return new, new_labels


def distribution(labels, name, plt=None):
    ones = 0
    zeros = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            zeros += 1
        else:
            ones += 1
    print(ones)
    print(zeros)
    # equal number of ones and zeros: 15000


train = pd.read_csv('drive/MyDrive/researchProject/researchProject/DataSet/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

reformat_samples, reformat_lables = reformat(train.review, train.sentiment)

train_samples, test_samples, train_lables, test_lables = train_test_split(reformat_samples, reformat_lables, test_size=0.2, random_state=42)

train_samples = np.array(train_samples)
train_samples = np.expand_dims(train_samples, axis=-1)
test_samples = np.array(test_samples)
test_samples = np.expand_dims(test_samples, axis=-1)
train_lables = np.array(train_lables)
test_lables = np.array(test_lables)

num_labels = 2
image_size = 32
num_channels = 1