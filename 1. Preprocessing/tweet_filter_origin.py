import pandas as pd
import re, os
import matplotlib.pyplot as plt
from collections import Counter
import ast
import logging

logging.basicConfig(level=logging.INFO, filename="log-filter_origin.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter_origin.txt ...")
additional_stop_words = []
src_file_path = "./Filterd_Stream_20200807/"
filter_path = "./"
dst_file_path = filter_path + "/Filterd_Stream_20200807_filter_origin/"
FIGURE_PATH = dst_file_path
DATA_PATH = dst_file_path
ORIG_TWEET_CONTENT_FILE = DATA_PATH + 'all_tweets.csv'
ORIG_TWEET_FILE = DATA_PATH + 'origin_tweets.csv'
RE_TWEET_FILE = DATA_PATH + 'retweets.csv'
TRANS_TWEET_FILE = DATA_PATH + 'all_trans_tweets.csv'
LANG_CSV_FILE = DATA_PATH + 'language_count.csv'
DATE_STATS_CSV_FILE = DATA_PATH + 'date_stats.csv'
LANG_COUNT_FIG_FILE = FIGURE_PATH + 'lang_count.png'

logging.info("begin filtering " + src_file_path)

if not os.path.exists(filter_path):
    os.mkdir(filter_path)

if not os.path.exists(FIGURE_PATH):
    os.mkdir(FIGURE_PATH)

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

def get_all_tweets():
    '''
    Extract all tweets or load from a saved file
    '''
    if os.path.isfile(ORIG_TWEET_CONTENT_FILE):
        #tweets_df = read_data_from_pickle(ORIG_TWEET_FILE)
        tweets_df = pd.read_csv(ORIG_TWEET_CONTENT_FILE)
        lang_count = pd.read_csv(LANG_CSV_FILE)
        print('Loaded tweet extracts from file\n')
    else:
        print('Start reading tweets from twitter-sample files...\n')
        if not os.path.exists(src_file_path):
            print("no file in {}.".format(src_file_path))
            return None

        file_object = open(ORIG_TWEET_FILE, 'a', encoding='utf8')

        text_list = []
        for root, dirs, files in os.walk(src_file_path):
            for file in files:
                if file.endswith('.csv'):
                    logging.warning(file)
                    total_count = 0
                    eng_count = 0
                    origin_count = 0
                    retweets_count = 0
                    for line in open(os.path.join(src_file_path, file), 'r', encoding='utf-8'):
                        total_count += 1
                        record = ast.literal_eval(line)  # json.dumps
                        if 'data' in record:
                            if record['data']['lang'] != 'en':
                                continue
                            eng_count += 1

                            if 'text' in record['data']:
                                #get original tweets
                                if 'in_reply_to_user_id' in record['data'] or 'referenced_tweets' in record['data']:
                                    #text = re.sub(r'[\r\n]|(\w+:\/\/\S+)', ' ', record['data']['text'])
                                    #text_list.append([record['data']['id'], text, record['data']['lang'],
                                    #                  record['data']['created_at'][:10]])
                                    retweets_count += 1
                                    continue

                                #text = re.sub(r'[\r\n]|(\w+:\/\/\S+)', ' ', record['data']['text'])
                                #text_list.append([record['data']['id'], text, record['data']['lang'], record['data']['created_at'][:10]])
                                origin_count += 1
                                file_object.write("{}".format(line))

                print(
                    '{} total_count: {}, eng_count:{}, origin_count: {}, eng_percent:{:.2%}, orig_percent: {:.2%}, retweets_count: {}, percent: {:.2%}'.format(
                        file, total_count, eng_count, origin_count, eng_count / total_count, origin_count / eng_count,
                        retweets_count, retweets_count / eng_count))

                logging.info('{} total_count: {}, eng_count:{}, origin_count: {}, eng_percent:{:.2%}, orig_percent: {:.2%}, retweets_count: {}, percent: {:.2%}'.format(
                        file, total_count, eng_count, origin_count, eng_count / total_count, origin_count / eng_count,
                        retweets_count, retweets_count / eng_count))
        file_object.close()
        # Convert list of tweets to DataFrame
        tweets_df = pd.DataFrame(text_list, columns=["id", "tweet_text", "lang", "date"])
        """
        lang_frequency = Counter(tweets_df["lang"]).most_common()
        lang_count = pd.DataFrame(data=lang_frequency,
                                  columns=['language', 'frequency'])
        lang_count.to_csv(LANG_CSV_FILE, index=False)
        # Save tweet extracts to file
        """
        tweets_df.to_csv(RE_TWEET_FILE, index=False)
        #save_data_to_pickle(ORIG_TWEET_FILE, tweets_df)

        print('Tweet extracts saved\n')

    date_stats = []
    for date in set(tweets_df['date']):
        daily_df = tweets_df[(tweets_df['date']==date)]
        daily_count = daily_df.shape[0]
        en_count = daily_df[daily_df['lang'] == 'en'].shape[0]
        es_count = daily_df[daily_df['lang'] == 'es'].shape[0]
        pt_count = daily_df[daily_df['lang'] == 'pt'].shape[0]
        hi_count = daily_df[daily_df['lang'] == 'hi'].shape[0]
        ja_count = daily_df[daily_df['lang'] == 'ja'].shape[0]
        date_stats.append(
            [date, daily_df.shape[0], format(en_count / daily_count, '.0%'), format(es_count / daily_count, '.0%'),
             format(pt_count / daily_count, '.0%'), format(hi_count / daily_count, '.0%'),
             format(ja_count / daily_count, '.0%')])

    date_stats_df = pd.DataFrame(data=date_stats, columns=['date', 'count', 'en', 'es', 'pt', 'hi', 'ja'])
    date_stats_df.to_csv(DATE_STATS_CSV_FILE, index=False)

    lang_count = lang_count.head(12)
    explode = [x * 0.05 for x in range(lang_count.shape[0])]  # 与labels一一对应，数值越大离中心区越远
    plt.axes(aspect=1)  # 设置X轴 Y轴比例
    # labeldistance标签离中心距离  pctdistance百分百数据离中心区距离 autopct 百分比的格式 shadow阴影
    plt.pie(x=lang_count['frequency'], labels=lang_count['language'], explode=explode, autopct='%3.2f %%',
            shadow=False, labeldistance=1.1, startangle=0, pctdistance=0.8, center=(-1, 0))
    # 控制位置：bbox_to_anchor数组中，前者控制左右移动，后者控制上下。ncol控制 图例所列的列数。默认值为1。fancybox 圆边
    plt.legend(loc=7, bbox_to_anchor=(1.23, 0.80), ncol=3, fancybox=True, shadow=True, fontsize=8)
    #plt.show()
    plt.savefig(LANG_COUNT_FIG_FILE)

    return tweets_df


if __name__ == '__main__':
    # Get all tweets
    all_tweets_df = get_all_tweets()

    print('DONE!')
