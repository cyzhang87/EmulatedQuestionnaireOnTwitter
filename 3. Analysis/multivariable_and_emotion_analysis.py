import statsmodels.api as sm
import numpy as np
import pandas as pd
from  scipy.stats import chi2_contingency

print("************ univariate analysis: 1 variables **************")
#人群属性与关注领域
import pandas as pd
files = [
         './data/origin_filter_0807/tweets_analysis_result.csv',
         './data/two_filter_0807/tweets_analysis_result_plutchik_onehot_1104.csv']
         #'./data/two_filter_0807/tweets_analysis_result.csv']

nodules_list_df = pd.read_csv(files[0])
origin_df = nodules_list_df[["age", "org", "senti-score", "gender", "drug_label", "eco_label", "pol_label", "edu_label", "ent_label"]]
nodules_list_df = pd.read_csv(files[1])
covid_df = nodules_list_df[["age", "org", "senti-score", "gender", "drug_label", "eco_label", "pol_label", "edu_label", "ent_label"]]

origin_person_df = origin_df[origin_df["org"] == 0]
covid_person_df = covid_df[covid_df["org"] == 0]
origin_org_df = origin_df[origin_df["org"] == 1]
covid_org_df = covid_df[covid_df["org"] == 1]

def calculate_forest_OR(or_table):
    result = sm.stats.Table2x2(or_table)
    """
    oddsratio = round(result.oddsratio, 2)
    lcb = round(result.oddsratio_confint()[0], 2)
    ucb = round(result.oddsratio_confint()[1], 2)
    ratio = round(or_table[0][0] / (or_table[0][0] + or_table[0][1]) * 100, 2)
    """
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    ratio = or_table[0][0] / (or_table[0][0] + or_table[0][1]) * 100

    return ["{:.2f}".format(ratio), "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb), '']

result_list = []
#user type
print("1 variables \r\nuser type: ")
table = [[len(covid_person_df), 913480],
         [len(covid_org_df), 102175]]
result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
or_table = [[table[1][0], table[1][1] - table[1][0]],
            [table[0][0], table[0][1] - table[0][0]]]
result_list.append(calculate_forest_OR(or_table))
print(np.array(or_table))
print(result.summary(method='normal'))
result_list.append([''] * 6)
#gender
print("gender: ")
origin_male_df = origin_person_df[origin_person_df["gender"] == 0]
covid_male_df = covid_person_df[covid_person_df["gender"] == 0]
origin_female_df = origin_person_df[origin_person_df["gender"] == 1]
covid_female_df = covid_person_df[covid_person_df["gender"] == 1]

table = [[len(covid_male_df), 481770],
         [len(covid_female_df), 431710]]
result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
or_table = [[table[1][0], table[1][1] - table[1][0]],
            [table[0][0], table[0][1] - table[0][0]]]
result_list.append(calculate_forest_OR(or_table))
print(np.array(or_table))
print(result.summary(method='normal'))
result_list.append([''] * 6)
#age
print("gender: ")
age_num = [346483, 350959, 104228, 111810]
table = []
for i in range(len(age_num)):
    covid_age_df = covid_person_df[covid_person_df["age"] == i]
    table.append([len(covid_age_df), age_num[i]])

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(age_num)-1):
    print("age: " + str(i+1))
    or_table = [[table[i+1][0], table[i+1][1] - table[i+1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))
result_list.append([''] * 6)
#label
print("label: ")
label_name = ["eco_label", "drug_label", "pol_label", "edu_label", "ent_label"]
label_num = [142090, 141176, 73838, 64799, 79119]
table = []
for i in range(len(label_num)):
    covid_label_df = covid_df[covid_df[label_name[i]] == 1]
    table.append([len(covid_label_df), label_num[i]])

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(label_num)-1):
    print("label: " + label_name[i])
    or_table = [[table[i+1][0], table[i+1][1] - table[i+1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))

sentiment_result = []
df_list = [covid_person_df, covid_org_df, '', covid_male_df, covid_female_df, '']
for i in range(4):
    df_list.append(covid_person_df[covid_person_df["age"] == i])
df_list.append('')
for i in range(len(label_name)):
    df_list.append(covid_df[covid_df[label_name[i]] == 1])
for df in df_list:
    if type(df) == type(''):
        sentiment_result.append('')
        continue
    mean = np.mean(df["senti-score"])
    sentiment_result.append("{:.4f}".format(mean))

for i in range(len(result_list)):
    result_list[i][5] = sentiment_result[i]

result_df = pd.DataFrame(columns=['attention ratio', 'odd_ratio', 'lcb', 'ucb', 'ci', 'sentiment'], data=result_list)
result_df.to_csv("1viables_odds_ratio_0807.csv", index=False)

print("************ bivariate analysis: 2 variables **************")
result_list = []
#age gender
table = []
df_list = []
print("age * gender: ")
age_gender_num = []
for i in range(len(age_num)):
    covid_age_df = covid_person_df[covid_person_df["age"] == i]
    origin_age_df = origin_person_df[origin_person_df["age"] == i]
    covid_age_male_df = covid_age_df[covid_age_df["gender"] == 0]
    covid_age_female_df = covid_age_df[covid_age_df["gender"] == 1]
    age_gender_num.append([len(origin_age_df[origin_age_df["gender"] == 0]),
                           len(origin_age_df[origin_age_df["gender"] == 1])])
    table.append([len(covid_age_male_df), len(origin_age_df[origin_age_df["gender"] == 0])])
    table.append([len(covid_age_female_df), len(origin_age_df[origin_age_df["gender"] == 1])])
    df_list.append(covid_age_male_df)
    df_list.append(covid_age_female_df)
df_list.append('')

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(table)-1):
    or_table = [[table[i+1][0], table[i+1][1] - table[i+1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))
result_list.append([''] * 6)

print("user type * label: ")
origin_person_df = origin_df[origin_df["org"] == 0]
covid_person_df = covid_df[covid_df["org"] == 0]
origin_org_df = origin_df[origin_df["org"] == 1]
covid_org_df = covid_df[covid_df["org"] == 1]
table = [[len(covid_person_df[covid_person_df["eco_label"] == 1]),  len(origin_person_df[origin_person_df["eco_label"] == 1])],  #person, eco
         [len(covid_person_df[covid_person_df["drug_label"] == 1]), len(origin_person_df[origin_person_df["drug_label"] == 1])], #person, drug
         [len(covid_person_df[covid_person_df["pol_label"] == 1]),  len(origin_person_df[origin_person_df["pol_label"] == 1])],  #person, pol
         [len(covid_person_df[covid_person_df["edu_label"] == 1]),  len(origin_person_df[origin_person_df["edu_label"] == 1])],  #person, edu
         [len(covid_person_df[covid_person_df["ent_label"] == 1]),  len(origin_person_df[origin_person_df["ent_label"] == 1])],  #person, ent
         [len(covid_org_df[covid_org_df["eco_label"] == 1]),  len(origin_org_df[origin_org_df["eco_label"] == 1])],  #org, eco
         [len(covid_org_df[covid_org_df["drug_label"] == 1]), len(origin_org_df[origin_org_df["drug_label"] == 1])], #org, drug
         [len(covid_org_df[covid_org_df["pol_label"] == 1]),  len(origin_org_df[origin_org_df["pol_label"] == 1])],  #org, pol
         [len(covid_org_df[covid_org_df["edu_label"] == 1]),  len(origin_org_df[origin_org_df["edu_label"] == 1])],  #org, edu
         [len(covid_org_df[covid_org_df["ent_label"] == 1]),  len(origin_org_df[origin_org_df["ent_label"] == 1])]]  #org, ent

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(table) - 1):
    or_table = [[table[i + 1][0], table[i + 1][1] - table[i + 1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))
result_list.append([''] * 6)

for i in range(len(label_name)):
    df_list.append(covid_person_df[covid_person_df[label_name[i]] == 1])
for i in range(len(label_name)):
    df_list.append(covid_org_df[covid_org_df[label_name[i]] == 1])
df_list.append('')

print("gender * label: ")
table = []
gender_num = [481770, 431710]
for i in range(2):
    origin_gender_df = origin_person_df[origin_person_df["gender"] == i]
    covid_gender_df = covid_person_df[covid_person_df["gender"] == i]
    for j in range(len(label_name)):
        table.append([len(covid_gender_df[covid_gender_df[label_name[j]] == 1]), len(origin_gender_df[origin_gender_df[label_name[j]] == 1])])

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(table) - 1):
    or_table = [[table[i + 1][0], table[i + 1][1] - table[i + 1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))
result_list.append([''] * 6)

for i in range(2):
    covid_gender_df = covid_person_df[covid_person_df["gender"] == i]
    for j in range(len(label_name)):
        df_list.append(covid_gender_df[covid_gender_df[label_name[j]] == 1])
df_list.append('')

print("age * label: ")
table = []
for i in range(4):
    covid_age_df = covid_person_df[covid_person_df["age"] == i]
    origin_age_df = origin_person_df[origin_person_df["age"] == i]
    for j in range(len(label_name)):
        origin_label_df = origin_age_df[origin_age_df[label_name[j]] == 1]
        covid_label_df = covid_age_df[covid_age_df[label_name[j]] == 1]
        table.append([len(covid_label_df), len(origin_label_df)])

result_list.append([round(table[0][0] / table[0][1] * 100, 2), '', '', '', '1 (ref)', ''])
for i in range(len(table) - 1):
    or_table = [[table[i + 1][0], table[i + 1][1] - table[i + 1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print(np.array(or_table))
    print(result.summary(method='normal'))

for i in range(4):
    covid_age_df = covid_person_df[covid_person_df["age"] == i]
    for j in range(len(label_name)):
        df_list.append(covid_age_df[covid_age_df[label_name[j]] == 1])

sentiment_result = []
for df in df_list:
    if type(df) == type(''):
        sentiment_result.append('')
        continue
    mean = np.mean(df["senti-score"])
    sentiment_result.append("{:.4f}".format(mean))

for i in range(len(result_list)):
    result_list[i][5] = sentiment_result[i]

result_df = pd.DataFrame(columns=['attention ratio', 'odd_ratio', 'lcb', 'ucb', 'ci', 'sentiment'], data=result_list)
result_df.to_csv("2viables_odds_ratio_0807.csv", index=False)

print("************trivariate analysis: 3 variables **************")
# male age five label
label_name = ["eco_label", "drug_label", "pol_label", "edu_label", "ent_label"]
table = []
df_list = []

for k in range(2):
    for i in range(4):
        if k == 0:
            origin_age_df = origin_male_df[origin_male_df["age"] == i]
            covid_age_df = covid_male_df[covid_male_df["age"] == i]
        else:
            origin_age_df = origin_female_df[origin_female_df["age"] == i]
            covid_age_df = covid_female_df[covid_female_df["age"] == i]

        for j in range(len(label_name)):
            table.append([len(covid_age_df[covid_age_df[label_name[j]] == 1]), len(origin_age_df[origin_age_df[label_name[j]] == 1])])
            df_list.append(covid_age_df[covid_age_df[label_name[j]] == 1])

result_list = []
result_list.append(["{:.2f}".format(table[0][0] / table[0][1] * 100), '', '', '', '1 (ref)', ''])
for i in range(len(table)-1):
    or_table = [[table[i + 1][0], table[i + 1][1] - table[i + 1][0]],
                [table[0][0], table[0][1] - table[0][0]]]
    result_list.append(calculate_forest_OR(or_table))
    print("gender*age*type data: {}".format(i))
    print(np.array(or_table))
    print(result.summary(method='normal'))

sentiment_result = []
for df in df_list:
    if type(df) == type(''):
        sentiment_result.append('')
        continue
    mean = np.mean(df["senti-score"])
    sentiment_result.append("{:.4f}".format(mean))

for i in range(len(result_list)):
    result_list[i][5] = sentiment_result[i]

result_df = pd.DataFrame(columns=['attention ratio', 'odd_ratio', 'lcb', 'ucb', 'ci', 'sentiment'], data=result_list)
result_df.to_csv("3viables_odds_ratio_0807.csv", index=False)

print("############### Plutchik emotion model ######################")
def calculate_OR(or_talbe, emotion_label):
    result = sm.stats.Table2x2(or_talbe)
    oddsratio = round(result.oddsratio, 2)
    lcb = round(result.oddsratio_confint()[0], 2)
    ucb = round(result.oddsratio_confint()[1], 2)
    pvalue = result.oddsratio_pvalue()
    if pvalue < 0.001:
        pvalue_str = '<0.001'
    else:
        pvalue_str = str(round(pvalue, 3))
    #print(np.array(or_talbe))
    #print(result.summary(method='normal'))
    ratio = round(or_talbe[0][0]/(or_talbe[0][0]+or_talbe[0][1])*100, 2)
    return [emotion_label, ratio, oddsratio, lcb, ucb, str(oddsratio) + ' (' + str(lcb) + '-' + str(ucb) + ')', pvalue_str, '']


from itertools import combinations

def calculate_chi2(table):
    kf = chi2_contingency(table)
    pvalue = kf[1]
    if pvalue < 0.001:
        pvalue_str = '<0.001'
    else:
        pvalue_str = str(round(pvalue, 3))


    #改成平均卡方
    pvalue2 = 0
    count = 0
    tableT = np.transpose(table)
    for i in combinations(tableT, 2):
        kf = chi2_contingency([i[0], i[1]])
        pvalue2 += kf[1]
        count += 1
    pvalue2 = pvalue2 / count
    if pvalue2 < 0.001:
        pvalue2_str = '<0.001'
    else:
        pvalue2_str = str(round(pvalue2, 3))

    return ([''] * 7 + [pvalue2_str])

file = './data/two_filter_0807/tweets_analysis_result_plutchik_onehot_1102.csv'
#file = './data/two_filter_0807/tweets_analysis_result_plutchik_onehot_1104.csv'

covid_df = pd.read_csv(file)
covid_person_df = covid_df[covid_df["org"] == 0]
covid_org_df = covid_df[covid_df["org"] == 1]
covid_male_df = covid_person_df[covid_person_df["gender"] == 0]
covid_female_df = covid_person_df[covid_person_df["gender"] == 1]
emotion_labels = ['Fear', 'Trust', 'Joy', 'Anticipation', 'Anger', 'Disgust', 'Sadness', 'Surprise']

result_list = []

for i in range(len(emotion_labels)):
    print(emotion_labels[i])
    print("emotion type:")
    or_table = [[len(covid_org_df[covid_org_df[emotion_labels[i]] == 1]), len(covid_org_df[covid_org_df[emotion_labels[i]] == 0])],
                [len(covid_person_df[covid_person_df[emotion_labels[i]] == 1]), len(covid_person_df[covid_person_df[emotion_labels[i]] == 0])]]
    result_list.append([emotion_labels[i], round(or_table[1][0] / (or_table[1][0] + or_table[1][1]) * 100, 2), '', '', '', '1 (ref)', '', ''])
    result_list.append(calculate_OR(or_table, emotion_labels[i]))
    result_list.append(calculate_chi2(or_table))

    print("emotion gender:")
    or_table = [[len(covid_female_df[covid_female_df[emotion_labels[i]] == 1]),
                 len(covid_female_df[covid_female_df[emotion_labels[i]] == 0])],
                [len(covid_male_df[covid_male_df[emotion_labels[i]] == 1]),
                 len(covid_male_df[covid_male_df[emotion_labels[i]] == 0])]]
    result_list.append(
        [emotion_labels[i], round(or_table[1][0] / (or_table[1][0] + or_table[1][1]) * 100, 2), '', '', '', '1 (ref)', '',
         ''])
    result_list.append(calculate_OR(or_table, emotion_labels[i]))
    result_list.append(calculate_chi2(or_table))

    print("emotion ages:")
    table = []
    for j in range(4):
        covid_age_df = covid_person_df[covid_person_df["age"] == j]
        table.append([len(covid_age_df[covid_age_df[emotion_labels[i]] == 1]), len(covid_age_df[covid_age_df[emotion_labels[i]] == 0])])

    result_list.append(
        [emotion_labels[i], round(table[0][0] / (table[0][0] + table[0][1])*100, 2), '', '', '', '1 (ref)', '',
         ''])

    for j in range(3):
        or_table = [table[j + 1], table[0]]
        result_list.append(calculate_OR(or_table, emotion_labels[i]))

    tableT = np.transpose(np.array(table))
    result_list.append(calculate_chi2(tableT))

    print("emotion topic:")
    table = []
    for j in range(len(label_name)):
        covid_topic_df = covid_df[covid_df[label_name[j]] == 1]
        table.append([len(covid_topic_df[covid_topic_df[emotion_labels[i]] == 1]),
                      len(covid_topic_df[covid_topic_df[emotion_labels[i]] == 0])])

    result_list.append(
        [emotion_labels[i], round(table[0][0] / (table[0][0] + table[0][1])*100, 2), '', '', '', '1 (ref)', '',
         ''])
    for j in range(len(label_name) - 1):
        or_table = [table[j + 1], table[0]]
        result_list.append(calculate_OR(or_table, emotion_labels[i]))

    tableT = np.transpose(np.array(table))
    result_list.append(calculate_chi2(tableT))

    result_list.append([''] * 7)

result_df = pd.DataFrame(columns=['emotion type', 'ratio','odd_ratio', 'lcb', 'ucb', 'ci', 'p-value', 'chi2-pvalue'], data=result_list)
result_df.to_csv("./data/two_filter_0807/plutchik_emotion_odds_ratio.csv", index=False)

print("############### Poms emotion model ######################")
file = './data/two_filter_0807/tweets_analysis_result_poms_onehot_1102.csv'
#file = './data/two_filter_0807/tweets_analysis_result_poms_onehot_1104.csv'

covid_df = pd.read_csv(file)
covid_person_df = covid_df[covid_df["org"] == 0]
covid_org_df = covid_df[covid_df["org"] == 1]
covid_male_df = covid_person_df[covid_person_df["gender"] == 0]
covid_female_df = covid_person_df[covid_person_df["gender"] == 1]
emotion_labels = ['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']

result_list = []

for i in range(len(emotion_labels)):
    print(emotion_labels[i])
    print("emotion type:")
    or_table = [[len(covid_org_df[covid_org_df[emotion_labels[i]] == 1]), len(covid_org_df[covid_org_df[emotion_labels[i]] == 0])],
                [len(covid_person_df[covid_person_df[emotion_labels[i]] == 1]), len(covid_person_df[covid_person_df[emotion_labels[i]] == 0])]]
    result_list.append(
        [emotion_labels[i], round(or_table[1][0] / (or_table[1][0] + or_table[1][1]) * 100, 2), '', '', '', '1 (ref)', '',
         ''])
    result_list.append(calculate_OR(or_table, emotion_labels[i]))
    result_list.append(calculate_chi2(or_table))

    print("emotion gender:")
    or_table = [[len(covid_female_df[covid_female_df[emotion_labels[i]] == 1]),
                 len(covid_female_df[covid_female_df[emotion_labels[i]] == 0])],
                [len(covid_male_df[covid_male_df[emotion_labels[i]] == 1]),
                 len(covid_male_df[covid_male_df[emotion_labels[i]] == 0])]]
    result_list.append(
        [emotion_labels[i], round(or_table[1][0] / (or_table[1][0] + or_table[1][1]) * 100, 2), '', '', '', '1 (ref)', '',
         ''])
    result_list.append(calculate_OR(or_table, emotion_labels[i]))
    result_list.append(calculate_chi2(or_table))

    print("emotion ages:")
    table = []
    for j in range(4):
        covid_age_df = covid_person_df[covid_person_df["age"] == j]
        table.append([len(covid_age_df[covid_age_df[emotion_labels[i]] == 1]), len(covid_age_df[covid_age_df[emotion_labels[i]] == 0])])

    result_list.append(
        [emotion_labels[i], round(table[0][0] / (table[0][0] + table[0][1])*100, 2), '', '', '', '1 (ref)', '',
         ''])

    for j in range(3):
        or_table = [table[j + 1], table[0]]
        result_list.append(calculate_OR(or_table, emotion_labels[i]))

    tableT = np.transpose(np.array(table))
    result_list.append(calculate_chi2(tableT))

    print("emotion topic:")
    table = []
    for j in range(len(label_name)):
        covid_topic_df = covid_df[covid_df[label_name[j]] == 1]
        table.append([len(covid_topic_df[covid_topic_df[emotion_labels[i]] == 1]),
                      len(covid_topic_df[covid_topic_df[emotion_labels[i]] == 0])])
    result_list.append(
        [emotion_labels[i], round(table[0][0] / (table[0][0] + table[0][1])*100, 2), '', '', '', '1 (ref)', '',
         ''])
    for j in range(len(label_name) - 1):
        or_table = [table[j + 1], table[0]]
        result_list.append(calculate_OR(or_table, emotion_labels[i]))

    tableT = np.transpose(np.array(table))
    result_list.append(calculate_chi2(tableT))


result_df = pd.DataFrame(columns=['emotion type', 'ratio', 'odd_ratio', 'lcb', 'ucb', 'ci', 'p-value', 'chi2-pvalue'], data=result_list)
result_df.to_csv("./data/two_filter_0807/poms_emotion_odds_ratio.csv", index=False)
