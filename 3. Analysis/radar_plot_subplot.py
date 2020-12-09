# 导入第三方模块
import numpy as np
import matplotlib.pyplot as plt
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 20}
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 15}
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格
plt.style.use('ggplot')
red_rgb = '#E24A33'
blue_rgb = '#348ABD'
# 构造数据
values0 = [[0.0987, 0.2460, 0.0458, 0.0121, 0.0891, 0.1907],
          [0.2069, 0.2691, 0.0457, 0.0181, 0.0484, 0.1332]]
feature0 = ['Fear', 'Joy', 'Anger', 'Disgust', 'Sadness', 'Surprise']
values1 = [[0.0687, 0.3267, 0.2758, 0.0184, 0.0314, 0.0044, 0.0777, 0.1968],
          [0.1978, 0.2559, 0.3153, 0.0145, 0.0278, 0.0092, 0.0423, 0.1372]]
feature1 = ['Fear', 'Trust', 'Joy', 'Anticipation', 'Anger', 'Disgust', 'Sadness', 'Surprise']
values2 = [[0.1577, 0.4871, 0.0629, 0.0522, 0.0924, 0.1477],
          [0.1225, 0.5734, 0.0523, 0.0777, 0.0780, 0.0962]]
feature2 = ['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']
title = ['Ekman Emotion Model',
         'Plutchik Emotion Model',
         'POMS Emotion Model']
ylim = [0.3, 0.4, 0.6]

axes = [''] * 3
for i in range(3):
    values = eval('values' + str(i))
    feature = eval('feature' + str(i))
    N = len(feature)
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    value0 = np.concatenate((values[0], [values[0][0]]))
    value1 = np.concatenate((values[1], [values[1][0]]))
    angle = np.concatenate((angles,[angles[0]]))
    fig_temp = plt.figure()
    # 这里一定要设置为极坐标格式
    axes[i] = fig_temp.add_subplot(111, polar=True)
    # 绘制折线图
    axes[i].plot(angle, value0, color=blue_rgb, linestyle='-', marker='o', linewidth=2, label="Original Tweets")
    axes[i].plot(angle, value1, color=red_rgb, linestyle='-', marker='o', linewidth=2, label="COVID-19 Tweets")
    # 填充颜色
    axes[i].fill(angle, value0, alpha=0.25, color=blue_rgb)
    axes[i].fill(angle, value1, alpha=0.25, color=red_rgb)
    # 添加每个特征的标签
    axes[i].set_thetagrids(angle * 180/np.pi, feature)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=13)
    # 设置雷达图的范围
    axes[i].set_ylim(0, ylim[i])
    # 添加标题
    plt.title(title[i], fontdict=font1)
    # 添加网格线
    axes[i].grid(True)
    if i == 0:
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fancybox=True, shadow=False, prop=font2)
    figure_path = 'radio_plot' + str(i) +'.pdf'
    plt.savefig(figure_path, bbox_inches='tight')
    plt.show()


# 显示图形

#fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = [''] * 3
fig_temp=plt.figure(figsize=(18, 6))
for i in range(3):
    values = eval('values' + str(i))
    feature = eval('feature' + str(i))
    N = len(feature)
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    value0 = np.concatenate((values[0], [values[0][0]]))
    value1 = np.concatenate((values[1], [values[1][0]]))
    angle = np.concatenate((angles,[angles[0]]))

    # 这里一定要设置为极坐标格式
    axes[i] = fig_temp.add_subplot(eval('13' +str(i+1)), polar=True)
    # 绘制折线图
    axes[i].plot(angle, value0, color=blue_rgb, linestyle='-', marker='o', linewidth=2, label="Original Tweets")
    axes[i].plot(angle, value1, color=red_rgb, linestyle='-', marker='o', linewidth=2, label="COVID-19 Tweets")
    # 填充颜色
    axes[i].fill(angle, value0, alpha=0.25, color=blue_rgb)
    axes[i].fill(angle, value1, alpha=0.25, color=red_rgb)
    # 添加每个特征的标签
    axes[i].set_thetagrids(angle * 180/np.pi, feature)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=13)
    # 设置雷达图的范围
    axes[i].set_ylim(0,ylim[i])
    # 添加标题
    plt.title(title[i], fontdict=font1)
    # 添加网格线
    axes[i].grid(True)
plt.legend(loc='center', bbox_to_anchor=(-1, -0.2), ncol=2,frameon=False, fancybox=True, shadow=False, prop=font2)
plt.subplots_adjust(wspace=0.3)
# 显示图形
plt.savefig('radio_plot.pdf', bbox_inches='tight')
plt.show()

import pandas as pd
file = ['../data/two_filter_0807/tweets_analysis_result_plutchik_onehot_1104.csv',
        '../data/two_filter_0807/tweets_analysis_result_poms_onehot_1104.csv']
model_name = ['(Plutchik)', '(POMS)']
for f in range(2):
    #person org
    covid_df = pd.read_csv(file[f])
    covid_person_df = covid_df[covid_df["org"] == 0]
    covid_org_df = covid_df[covid_df["org"] == 1]
    covid_male_df = covid_person_df[covid_person_df["gender"] == 0]
    covid_female_df = covid_person_df[covid_person_df["gender"] == 1]
    feature = eval('feature' + str(f+1))
    df_list = [covid_person_df, covid_org_df]
    N = len(feature)
    values = np.zeros((2, N))
    for i in range(len(df_list)):
        for j in range(N):
            values[i][j] = np.mean(df_list[i][feature[j]])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    value0 = np.concatenate((values[0], [values[0][0]]))
    value1 = np.concatenate((values[1], [values[1][0]]))
    angle = np.concatenate((angles,[angles[0]]))

    # 绘图
    fig=plt.figure(figsize=(6, 6))
    # 这里一定要设置为极坐标格式
    ax = fig.add_subplot(221, polar=True)
    # 绘制折线图
    ax.plot(angle, value0, 'o-', linewidth=2, label="Person")
    ax.plot(angle, value1, 'o-', linewidth=2, label="Organization")
    # 填充颜色
    ax.fill(angle, value0, alpha=0.25)
    ax.fill(angle, value1, alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angle * 180/np.pi, feature)
    # 设置雷达图的范围
    ax.set_ylim(0,0.40)
    # 添加标题
    plt.title('User Type ' + model_name[f], fontsize=12)
    # 添加网格线
    ax.grid(True)
    # 显示图例
    plt.legend(loc=1,bbox_to_anchor=(2, 1), frameon=False)

    #gender
    df_list = [covid_male_df, covid_female_df]

    for i in range(len(df_list)):
        for j in range(N):
            values[i][j] = np.mean(df_list[i][feature[j]])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    value0 = np.concatenate((values[0], [values[0][0]]))
    value1 = np.concatenate((values[1], [values[1][0]]))
    angle = np.concatenate((angles,[angles[0]]))

    # 这里一定要设置为极坐标格式
    ax = fig.add_subplot(222, polar=True)
    # 绘制折线图
    ax.plot(angle, value0, 'o-', linewidth=2, label="Male")
    ax.plot(angle, value1, 'o-', linewidth=2, label="Female")
    # 填充颜色
    ax.fill(angle, value0, alpha=0.25)
    ax.fill(angle, value1, alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angle * 180/np.pi, feature)
    # 设置雷达图的范围
    ax.set_ylim(0,0.35)
    # 添加标题
    plt.title('Gender ' + model_name[f], fontsize=12)
    # 添加网格线
    ax.grid(True)
    # 显示图例
    plt.legend(loc=1,bbox_to_anchor=(1.8, 1), frameon=False)

    #age
    values = np.zeros((4, N))
    df_list = []
    for i in range(4):
        df_list.append(covid_person_df[covid_person_df["age"] == i])

    for i in range(len(df_list)):
        for j in range(N):
            values[i][j] = np.mean(df_list[i][feature[j]])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    value0 = np.concatenate((values[0], [values[0][0]]))
    value1 = np.concatenate((values[1], [values[1][0]]))
    value2 = np.concatenate((values[2], [values[2][0]]))
    value3 = np.concatenate((values[3], [values[3][0]]))
    angle = np.concatenate((angles,[angles[0]]))

    ax = fig.add_subplot(223, polar=True)
    # 绘制折线图
    ax.plot(angle, value0, 'o-', linewidth=2, label="<=18")
    ax.plot(angle, value1, 'o-', linewidth=2, label="19-29")
    ax.plot(angle, value2, 'o-', linewidth=2, label="30-39")
    ax.plot(angle, value3, 'o-', linewidth=2, label=">=40")
    # 填充颜色
    ax.fill(angle, value0, alpha=0.25)
    ax.fill(angle, value1, alpha=0.25)
    ax.fill(angle, value2, alpha=0.25)
    ax.fill(angle, value3, alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angle * 180/np.pi, feature)
    # 设置雷达图的范围
    ax.set_ylim(0,0.35)
    # 添加标题
    plt.title('Age ' + model_name[f], fontsize=12)
    # 添加网格线
    ax.grid(True)
    # 显示图例
    plt.legend(loc=1,bbox_to_anchor=(1.8, 1), frameon=False)


    #topic
    label_name = ["eco_label", "drug_label", "pol_label", "edu_label", "ent_label"]
    M = len(label_name)
    values = np.zeros((M, N))
    df_list = []
    for i in range(M):
        df_list.append(covid_df[covid_df[label_name[i]] == 1])

    for i in range(len(df_list)):
        for j in range(N):
            values[i][j] = np.mean(df_list[i][feature[j]])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)

    # 为了使雷达图一圈封闭起来，需要下面的步骤
    values_new = np.zeros((M, N+1))
    for i in range(M):
        values_new[i] = np.concatenate((values[i], [values[i][0]]))

    angle = np.concatenate((angles,[angles[0]]))

    ax = fig.add_subplot(224, polar=True)
    # 绘制折线图
    for i in range(M):
        ax.plot(angle, values_new[i], 'o-', linewidth=2, label=label_name[i])
        # 填充颜色
        ax.fill(angle, values_new[i], alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angle * 180/np.pi, feature)
    # 设置雷达图的范围
    ax.set_ylim(0,0.40)
    # 添加标题
    plt.title('Topic ' + model_name[f], fontsize=12)
    # 添加网格线
    ax.grid(True)
    plt.legend(loc=1,bbox_to_anchor=(2, 1), frameon=False)
    plt.subplots_adjust(wspace=1.5, hspace=0.3)
    plt.savefig('population_radio_plot' + str(f) + '.pdf', bbox_inches='tight')
    plt.show()
