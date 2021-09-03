import string
import wordcloud
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
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


def get_train_test(rows):
    # Use a breakpoint in the code line below to debug your script.
    train = pd.read_csv('drive/MyDrive/researchProject/researchProject/DataSet/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3, nrows=rows)
    X_train, X_test, y_train, y_test = train_test_split(train.review, train.sentiment, test_size=0.2)
    return X_train, y_train, X_test, y_test


def construct_train_dic(train_review, train_sentiment):
    train_review = train_review.tolist()
    train_sentiment = train_sentiment.tolist()
    length_train = len(train_sentiment)
    dic_word_freq_for_positive = {}
    positive_length = 0
    dic_word_freq_for_negative = {}
    negative_length = 0
    # print(len(train_review))
    # print(len(train_sentiment))
    # print(type(train_review))
    for n in range(length_train):
        # print(train_review[n])
        text = remove_html_tags(train_review[n])
        text = to_lowercase(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = remove_extra_whitespace_tabs(text)
        text = nltk.word_tokenize(text)
        filtered_words = [word for word in text if word not in stopwords.words('english')]
        if train_sentiment[n] == 0:
            for word in filtered_words:
                positive_length += 1
                if word in dic_word_freq_for_positive:
                    dic_word_freq_for_positive[word] += 1
                else:
                    dic_word_freq_for_positive[word] = 1
        else:
            for word in filtered_words:
                negative_length += 1
                if word in dic_word_freq_for_negative:
                    dic_word_freq_for_negative[word] += 1
                else:
                    dic_word_freq_for_negative[word] = 1

    print(positive_length)
    print(negative_length)
    # for key in dic_word_freq_for_positive.keys():
    #     dic_word_freq_for_positive[key] = dic_word_freq_for_positive.get(key) / positive_length
    # for key in dic_word_freq_for_negative.keys():
    #     dic_word_freq_for_negative[key] = dic_word_freq_for_negative.get(key) / negative_length
    print(dic_word_freq_for_positive)
    print(dic_word_freq_for_negative)
    return positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative


def compute_leplace_method(test_review, test_setiment, posi_len, nega_len, dic_posi, dic_nega):
    test_review = test_review.tolist()
    test_setiment = test_setiment.tolist()
    length_sentiment = len(test_setiment)
    correct = 0

    for n in range(length_sentiment):
        text = remove_html_tags(test_review[n])
        text = to_lowercase(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = remove_extra_whitespace_tabs(text)
        text = nltk.word_tokenize(text)
        filtered_words = [word for word in text if word not in stopwords.words('english')]
        dic = {}
        for word in filtered_words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
        m_positive_len = len(dic_posi)
        m_negative_len = len(dic_nega)

        for key in dic.keys():
            if dic_posi.get(key) is None:
                m_positive_len += 1
            if dic_nega.get(key) is None:
                m_negative_len += 1

        prob_set_word_positve = {}
        prop_set_word_negative = {}
        sum_posi = 0
        sum_nega = 0
        for key in dic.keys():
            if key in dic_posi.keys():
                prob_set_word_positve[key] = (dic_posi.get(key) + 1) / (posi_len + m_positive_len)
                sum_posi += prob_set_word_positve[key]
            else:
                prob_set_word_positve[key] = 1 / (posi_len + m_positive_len)
                sum_posi += prob_set_word_positve[key]

            if key in dic_nega.keys():
                prop_set_word_negative[key] = (dic_nega.get(key) + 1) / (nega_len + m_negative_len)
                sum_nega += prop_set_word_negative[key]
            else:
                prop_set_word_negative[key] = 1 / (nega_len + m_negative_len)
                sum_nega += prop_set_word_negative[key]

        for key in dic_posi.keys():
            if key not in dic.keys():
                prob_set_word_positve[key] = (dic_posi.get(key) + 1) / (posi_len + m_positive_len)
                sum_posi += prob_set_word_positve[key]

        for key in dic_nega.keys():
            if key not in dic.keys():
                prop_set_word_negative[key] = (dic_nega.get(key) + 1) / (nega_len + m_negative_len)
                sum_nega += prop_set_word_negative[key]

        for key in dic.keys():
            prob_set_word_positve[key] = prob_set_word_positve[key] / sum_posi
            prop_set_word_negative[key] = prop_set_word_negative[key] / sum_nega

        print(prob_set_word_positve)
        print(prop_set_word_negative)

        positive_value = 0
        negative_value = 0

        for key in dic.keys():
            positive_value += dic.get(key) * np.log(prob_set_word_positve.get(key))
            negative_value += dic.get(key) * np.log(prop_set_word_negative.get(key))

        answer = -1
        if positive_value > negative_value:
            answer = 0
        else:
            answer = 1

        if answer == test_setiment[n]:
            correct += 1

    print(correct / length_sentiment)
    return correct / length_sentiment


def compute_ney_essen_method(test_review, test_setiment, posi_len, nega_len, dic_posi, dic_nega):
    test_review = test_review.tolist()
    test_setiment = test_setiment.tolist()
    length_sentiment = len(test_setiment)
    correct = 0
    r_dic_posi = len(dic_posi)
    r_dic_nega = len(dic_nega)

    for n in range(length_sentiment):
        text = remove_html_tags(test_review[n])
        text = to_lowercase(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = remove_extra_whitespace_tabs(text)
        text = nltk.word_tokenize(text)
        filtered_words = [word for word in text if word not in stopwords.words('english')]

        dic = {}
        dic_posi2 = dic_posi.copy()
        dic_nega2 = dic_nega.copy()
        for word in filtered_words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
            if dic_posi2.get(word) is None:
                dic_posi2[word] = 0
            if dic_nega2.get(word) is None:
                dic_nega2[word] = 0

        m_posi = len(dic_posi2)
        m_nega = len(dic_nega2)

        for key in dic_posi2.keys():
            if dic_posi2.get(key) == 0:
                dic_posi2[key] = r_dic_posi / m_posi
            else:
                dic_posi2[key] = dic_posi2[key] - 1 + r_dic_posi / m_posi

        for key in dic_nega2.keys():
            if dic_nega2.get(key) == 0:
                dic_nega2[key] = r_dic_nega / m_nega
            else:
                dic_nega2[key] = dic_nega2[key] - 1 + r_dic_nega / m_nega

        sum_posi = 0
        sum_nega = 0
        for key in dic_posi2.keys():
            dic_posi2[key] = dic_posi2[key] / posi_len
            sum_posi += dic_posi2[key]

        for key in dic_nega2.keys():
            dic_nega2[key] = dic_nega2[key] / nega_len
            sum_nega += dic_nega2[key]

        for key in dic_posi2.keys():
            dic_posi2[key] = dic_posi2[key] / sum_posi

        for key in dic_nega2.keys():
            dic_nega2[key] = dic_nega2[key] / sum_nega

        theta_posi = 0
        theta_nega = 0
        for key in dic.keys():
            theta_posi += dic.get(key) * np.log(dic_posi2.get(key))
            theta_nega += dic.get(key) * np.log(dic_nega2.get(key))

        print(theta_posi)
        print(theta_nega)

        answer = -1
        if theta_posi > theta_nega:
            answer = 0
        else:
            answer = 1

        if answer == test_setiment[n]:
            correct += 1

    print(correct / length_sentiment)
    return correct / length_sentiment


def compute_witten_bell_method(test_review, test_setiment, posi_len, nega_len, dic_posi, dic_nega):
    test_review = test_review.tolist()
    test_setiment = test_setiment.tolist()
    length_sentiment = len(test_setiment)
    correct = 0
    r_dic_posi = len(dic_posi)
    r_dic_nega = len(dic_nega)

    for n in range(length_sentiment):
        text = remove_html_tags(test_review[n])
        text = to_lowercase(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = remove_extra_whitespace_tabs(text)
        text = nltk.word_tokenize(text)
        filtered_words = [word for word in text if word not in stopwords.words('english')]

        dic = {}
        dic_posi2 = dic_posi.copy()
        dic_nega2 = dic_nega.copy()
        for word in filtered_words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
            if dic_posi2.get(word) is None:
                dic_posi2[word] = 0
            if dic_nega2.get(word) is None:
                dic_nega2[word] = 0

        m_posi = len(dic_posi2)
        m_nega = len(dic_nega2)

        sum_posi = 0
        sum_nega = 0
        for key in dic_posi2.keys():
            if dic_posi2.get(key) == 0:
                dic_posi2[key] = (1 / (m_posi - r_dic_posi)) * (r_dic_posi / (posi_len + r_dic_posi))
            else:
                dic_posi2[key] = dic_posi2[key] / (posi_len + r_dic_posi)
            sum_posi += dic_posi2.get(key)

        for key in dic_nega2.keys():
            if dic_nega2.get(key) == 0:
                dic_nega2[key] = (1 / (m_nega - r_dic_nega)) * (r_dic_nega / (nega_len + r_dic_nega))
            else:
                dic_nega2[key] = dic_nega2[key] / (nega_len + r_dic_nega)
            sum_nega += dic_nega2.get(key)

        for key in dic_posi2.keys():
            dic_posi2[key] = dic_posi2[key] / sum_posi

        for key in dic_nega2.keys():
            dic_nega2[key] = dic_nega2[key] / sum_nega

        theta_posi = 0
        theta_nega = 0
        for key in dic.keys():
            theta_posi += dic.get(key) * np.log(dic_posi2.get(key))
            theta_nega += dic.get(key) * np.log(dic_nega2.get(key))

        print(theta_posi)
        print(theta_nega)

        answer = -1
        if theta_posi > theta_nega:
            answer = 0
        else:
            answer = 1

        if answer == test_setiment[n]:
            correct += 1

    print(correct / length_sentiment)
    return correct / length_sentiment


epoch = 5
recursive_choice = 25000
leplace_value = 0
ney_essen_value = 0
witten_bell_value = 0
X_train, y_train, X_test, y_test = get_train_test(recursive_choice)
positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative = construct_train_dic(X_train,
                                                                                                               y_train)
for n in range(epoch):
    leplace_value += compute_leplace_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative)
    ney_essen_value += compute_ney_essen_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative)
    witten_bell_value += compute_witten_bell_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive,
                             dic_word_freq_for_negative)

leplace_value = leplace_value / 5
ney_essen_value = ney_essen_value / 5
witten_bell_value = witten_bell_value / 5

print(leplace_value)
print(ney_essen_value)
print(witten_bell_value)


# leplace_value_list = []
# ney_essen_list = []
# witten_bell_list = []
# while recursive_choice < 25000:
#     X_train, y_train, X_test, y_test = get_train_test(recursive_choice)
#     positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative = construct_train_dic(X_train,
#                                                                                                                    y_train)
#     leplace_value = 0
#     ney_essen_value = 0
#     witten_bell_value = 0
#     for n in range(epoch):
#         leplace_value += compute_leplace_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative)
#         ney_essen_value += compute_ney_essen_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive, dic_word_freq_for_negative)
#         witten_bell_value += compute_witten_bell_method(X_test, y_test, positive_length, negative_length, dic_word_freq_for_positive,
#                                  dic_word_freq_for_negative)
#
#     leplace_value = leplace_value / 5
#     ney_essen_value = ney_essen_value / 5
#     witten_bell_value = witten_bell_value / 5
#     leplace_value_list.append(leplace_value)
#     ney_essen_list.append(ney_essen_value)
#     witten_bell_list.append(witten_bell_value)
#     recursive_choice += 2000
#
# print(leplace_value_list)
# print(ney_essen_list)
# print(witten_bell_list)