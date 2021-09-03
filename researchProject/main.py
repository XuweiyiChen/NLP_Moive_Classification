# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from math import floor

from sklearn.model_selection import train_test_split


def get_training_and_testing_sets(file_list):
    split = 0.9
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    train = pd.read_csv('DataSet/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    print(train.shape)
    # X_train, X_test, y_train, y_test = train_test_split(train.review, train.sentiment, test_size=0.1,
    #                                                     random_state=42)
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
