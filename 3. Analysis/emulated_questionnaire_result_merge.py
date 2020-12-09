import pandas as pd

dir_name = "./data/two_filter_0807/"
tweets_label_file = dir_name + 'tweets_labels.csv'
tweets_user_file = dir_name + 'user_pred.csv'
#new_tweets_user_file = './result/tweets_user_pred.csv'
tweets_sentiment_file = dir_name + 'tweets_sentiment_scores.csv'
tweets_emotion_file = dir_name + 'tweets_emotion_scores_poms_onehot.csv'
result_file = dir_name + 'tweets_analysis_result_poms_onehot_1104.csv'

#label_df = pd.read_csv(result_file)


user_df = pd.read_csv(tweets_user_file)
"""
new_user_df = pd.DataFrame()
for index, tweet in user_df.iterrows():
    if tweet['org'] == 1:
        tweet['gender'] = 2
        tweet['age'] = 4
    new_user_df = new_user_df.append(tweet)

new_user_df.to_csv(new_tweets_user_file, index=False)
"""
#sentiment_df = pd.read_csv(tweets_sentiment_file, usecols=['senti-score', 'senti-degree'])
sentiment_df = pd.read_csv(tweets_sentiment_file)
emotion_df = pd.read_csv(tweets_emotion_file)
label_df = pd.read_csv(tweets_label_file)
result_df = pd.concat([user_df, label_df, sentiment_df, emotion_df], axis=1)
result_df.to_csv(result_file, index=False)

