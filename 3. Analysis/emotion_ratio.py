
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 构建数据

file = ['../data/origin_filter_0807/tweets_analysis_result_poms_onehot_1102.csv',
        '../data/two_filter_0807/tweets_analysis_result_poms_onehot_1104.csv']

origin_df = pd.read_csv(file[0])
covid_df = pd.read_csv(file[1])

# 构建数据
x_data = ['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']
y_data = []
y_data_num = origin_df.shape[0]
y_data2= []
y_data2_num = covid_df.shape[0]

for i in range(len(x_data)):
    y_data.append(round(len(origin_df[origin_df[x_data[i]] == 1])/y_data_num, 2))
for i in range(len(x_data)):
    y_data2.append(round(len(covid_df[covid_df[x_data[i]] == 1])/y_data2_num, 2))

bar_width = 0.42
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
plt.bar(x=x_data, height=y_data, label='Orignal Tweets',
    color='steelblue', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data2,
    label='Covid19 Tweets', color='indianred', alpha=0.8, width=bar_width)
for x, y in enumerate(y_data):
    plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=9)
for x, y in enumerate(y_data2):
    plt.text(x + bar_width, y+0.025, '%.2f' % y, ha='center', va='top', fontsize=9)
# 设置标题
plt.title("POMS Emotion Analysis")
# 为两条坐标轴设置名称
plt.xlabel("Emotions")
plt.ylabel("Fraction of Tweets")
plt.xticks(rotation=20)
# 显示图例
plt.legend()
plt.show()

file = ['../data/origin_filter_0807/tweets_analysis_result_plutchik_onehot_1102.csv',
        '../data/two_filter_0807/tweets_analysis_result_plutchik_onehot_1104.csv']

origin_df = pd.read_csv(file[0])
covid_df = pd.read_csv(file[1])

# 构建数据
x_data = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'Anticipation']
y_data = []
y_data_num = origin_df.shape[0]
y_data2= []
y_data2_num = covid_df.shape[0]

for i in range(len(x_data)):
    y_data.append(round(len(origin_df[origin_df[x_data[i]] == 1])/y_data_num, 2))
for i in range(len(x_data)):
    y_data2.append(round(len(covid_df[covid_df[x_data[i]] == 1])/y_data2_num, 2))

bar_width = 0.42
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
plt.bar(x=x_data, height=y_data, label='Orignal Tweets',
    color='steelblue', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data2,
    label='Covid19 Tweets', color='indianred', alpha=0.8, width=bar_width)
for x, y in enumerate(y_data):
    plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=9)
for x, y in enumerate(y_data2):
    plt.text(x + bar_width, y+0.013, '%.2f' % y, ha='center', va='top', fontsize=9)
# 设置标题
plt.title("Plutchik Emotion Analysis")
# 为两条坐标轴设置名称
plt.xlabel("Emotions")
plt.ylabel("Fraction of Tweets")
plt.xticks(rotation=20)
# 显示图例
plt.legend()
plt.show()

file = ['../data/origin_filter_0807/tweets_analysis_result_ekman_onehot_1102.csv',
        '../data/two_filter_0807/tweets_analysis_result_ekman_onehot_1104.csv']

origin_df = pd.read_csv(file[0])
covid_df = pd.read_csv(file[1])

# 构建数据
x_data = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
y_data = []
y_data_num = origin_df.shape[0]
y_data2= []
y_data2_num = covid_df.shape[0]

for i in range(len(x_data)):
    y_data.append(round(len(origin_df[origin_df[x_data[i]] == 1])/y_data_num, 2))
for i in range(len(x_data)):
    y_data2.append(round(len(covid_df[covid_df[x_data[i]] == 1])/y_data2_num, 2))

bar_width = 0.42
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
plt.bar(x=x_data, height=y_data, label='Orignal Tweets',
    color='steelblue', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data2,
    label='Covid19 Tweets', color='indianred', alpha=0.8, width=bar_width)
for x, y in enumerate(y_data):
    plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=9)
for x, y in enumerate(y_data2):
    plt.text(x + bar_width, y+0.013, '%.2f' % y, ha='center', va='top', fontsize=9)
# 设置标题
plt.title("Ekman Emotion Analysis")
# 为两条坐标轴设置名称
plt.xlabel("Emotions")
plt.ylabel("Fraction of Tweets")
plt.xticks(rotation=20)
# 显示图例
plt.legend()
plt.show()
