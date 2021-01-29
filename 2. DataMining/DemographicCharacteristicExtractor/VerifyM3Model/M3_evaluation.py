import ast
import logging
import json
import wikipedia
import re, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
dir_name = "two_filter_0807"
pred_file_name = "./user_pred_sample_users_0807.csv"
true_file_name = "./sample_users_gender_age_0807_good.csv"
step1_output_file_name = "./user_gender_age_pred_truth_0807.csv"
step2_output_file_name = "./appendix2_groudtruth_vs_prediction_0807.csv"

def merge_pred_true(pred_file_name, true_file_name, output_file_name):
    df1 = pd.read_csv(pred_file_name)
    df2 = pd.read_csv(true_file_name)

    total_count = 0
    age_true_count = 0
    gender_true_count = 0
    org_true_count = 0
    person_count = 0
    result_list = []
    for index, row in df2.iterrows():
        result = df1[df1['author_id'] == row['id']].reset_index()
        if len(result) == 0:
            continue
        org_true = 0
        gender_true = -1
        age_true = -1

        if row['age'] <= 0:
            age_true = -1
        if row['age'] <= 18:
            age_true = 0
        elif row['age'] <= 29:
            age_true = 1
        elif row['age'] <= 39:
            age_true = 2
        elif row['age'] >= 40:
            age_true = 3

        if row['gender'] == 'male':
            gender_true = 0
            person_count += 1
        elif row['gender'] == 'female':
            gender_true = 1
            person_count += 1
        elif row['gender'] == 'organization':
            gender_true = -1
            org_true = 1
            age_true = -1

        if result.loc[0]['org'] == 1:
            gender_pred = -1
            age_pred = -1
        else:
            gender_pred = result.loc[0]['gender']
            age_pred = result.loc[0]['age']

        """
        gender_pred = result.loc[0]['gender']
        age_pred = result.loc[0]['age']
        """

        result_list.append(
            [result.loc[0]['author_id'], row['screen_name'], org_true, row['age'], gender_true, age_true, result.loc[0]['org'],
             gender_pred, age_pred])

        total_count += 1
        if org_true == result.loc[0]['org']:
            org_true_count += 1
        if age_true == result.loc[0]['age']:
            age_true_count += 1
        if gender_true == result.loc[0]['gender']:
            gender_true_count += 1

    print("total:{}, person:{}, org:{} / {}, gender:{} / {}, age:{} / {}".format(
        total_count, person_count,
        org_true_count, org_true_count / total_count,
        gender_true_count, gender_true_count / person_count,
        age_true_count, age_true_count / person_count))
    df = pd.DataFrame(columns=['id', 'screen_name', 'true_org', 'true_age', 'true_gender', 'true_age_r', 'pred_org', 'pred_gender', 'pred_age'], data=result_list)

    print("org:")
    print(accuracy_score(df['true_org'], df['pred_org']))
    print(f1_score(df['true_org'], df['pred_org'], average='macro'))

    classes = ["<=18", "19-29", "30-39", ">=40"]
    df2 = df[df['true_org'] != 1]
    df2 = df2[df2['pred_org'] != 1]
    con_matrix = confusion_matrix(df2['true_age_r'], df2['pred_age'])
    plotCM(classes, con_matrix, './test.png')

    print("gender:")
    print(accuracy_score(df2['true_gender'], df2['pred_gender']))
    print(f1_score(df2['true_gender'], df2['pred_gender'], average='macro'))

    print("age:")
    print(accuracy_score(df2['true_age_r'], df2['pred_age']))
    print(f1_score(df2['true_age_r'], df2['pred_age'], average='macro'))

    df.to_csv(output_file_name, index=False, encoding='utf-8')


def plotCM(classes, matrix, savname):
    """classes: a list of class names"""

    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum

    # plot
    plt.switch_backend('agg')
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')

    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)

    # save
    plt.savefig(savname)

def remove_username_info(inputfile, outputfile):
    df = pd.read_csv(inputfile)
    df = df[['id', 'true_org', 'true_gender', 'true_age', 'true_age_r', 'pred_org', 'pred_gender', 'pred_age']]
    df.to_csv(outputfile, index=False, encoding='utf-8')

if __name__ == "__main__":
    merge_pred_true(pred_file_name, true_file_name, step1_output_file_name)
    remove_username_info(step1_output_file_name, step2_output_file_name)