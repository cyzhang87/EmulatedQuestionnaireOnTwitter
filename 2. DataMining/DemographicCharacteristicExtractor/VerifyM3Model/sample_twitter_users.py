import ast
import logging
import json
import re, os
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

dir_name = "two_filter_0807"
src_file_path = "/data/twitter_data/origin_tweets/Sampled_Stream_detail_20200807_0812_origin"
#step1_input_file_name = "../twitter_cache/" + dir_name + "/origin_tweets.csv"
step1_output_file_name = "../twitter_cache/" + dir_name + "/sample_users_0807.csv"
step2_output_file_name = "./sample_users_gender_age_0807.csv"
gender_file_name = "./gender_words.csv"
count = 0

def build_match_word_regex():
    global male_word_pattern, female_word_pattern, org_word_pattern
    dict_df = pd.DataFrame()
    if os.path.isfile(gender_file_name):
        dict_df = pd.read_csv(gender_file_name)
    else:
        print("gender file is not exit.")
        exit()
    print('Generate word_pattern ...\n')
    male_word_list = []
    female_word_list = []
    org_word_list = []
    for index, word in dict_df.iterrows():
        if word['type'] == 'male':
            male_word_list.append(r'\b' + word['word'] + r'(?![\w-])')
        elif word['type'] == 'female':
            female_word_list.append(r'\b' + word['word'] + r'(?![\w-])')
        elif word['type'] == 'organization':
            org_word_list.append(r'\b' + word['word'] + r'(?![\w-])')

    male_word_pattern = re.compile('|'.join(male_word_list), re.IGNORECASE)
    female_word_pattern = re.compile('|'.join(female_word_list), re.IGNORECASE)
    org_word_pattern = re.compile('|'.join(org_word_list), re.IGNORECASE)


def get_userid(src_file_path, output_file):
    for root, dirs, files in os.walk(src_file_path):
        for file in files:
            if file.endswith('.csv'):
                logging.warning(file)

            with open(file, "r", encoding='utf-8') as fhIn:
                id_list = []
                output_list = []
                for line in fhIn:
                    if isinstance(line, str):
                        line = ast.literal_eval(line)  # to dict
                        followers_count = line['includes']['users'][0]['stats']['followers_count']
                        id = line['includes']['users'][0]['id']
                        if followers_count >= 0:
                            if id in id_list:
                                continue
                            else:
                                id_list.append(id)
                            output_list.append([id, line['includes']['users'][0]['name'], line['includes']['users'][0]['username'],
                                                'https://twitter.com/' + line['includes']['users'][0]['username']])

        df = pd.DataFrame(columns=['id', 'name', 'username','website'], data=output_list)
        df.to_excel(output_file)

def age_inference(str):
    match = re.search('[1-2][0-9]{3}', str[str.find('born '):])
    if match:
        age = 2020 - int(match.group(0))
        return age
    else:
        return -1

def gender_inference(wiki_str):
    global count
    if re.search(male_word_pattern, wiki_str):
        return 'male'
    elif re.search(female_word_pattern, wiki_str):
        return 'female'
    elif re.search(org_word_pattern, wiki_str):
        return 'organization'
    else:
        count += 1
        return 'none'

def get_sample_users(src_file_path, output_file):
    with open(output_file, "w") as fhOut:
        id_list = []
        for root, dirs, files in os.walk(src_file_path):
            for file in files:
                if file.endswith('.csv'):
                    logging.warning(file)
                else:
                    continue
                with open(os.path.join(src_file_path, file), "r", encoding='utf-8') as fhIn:
                    for line in fhIn:
                        if isinstance(line, str):
                            line_tmp = ast.literal_eval(line)  # to dict
                            des = line_tmp['includes']['users'][0]['description']
                            id = line_tmp['includes']['users'][0]['id']
                            text = line_tmp['data']['text']
                            if re.search(r'\b' +'born'+ '(?![\w-])', des):
                                age = age_inference(des)
                                if age > 0:
                                    if id in id_list:
                                        continue
                                    else:
                                        id_list.append(id)
                                        fhOut.write("{}\n".format(line))
                                        continue
                            if re.search(r'\b' + 'born' + '(?![\w-])', text):
                                age = age_inference(text)
                                if age > 0:
                                    if id in id_list:
                                        continue
                                    else:
                                        id_list.append(id)
                                        fhOut.write("{}\n".format(line))
                        else:
                            logging.warning(line)
                            return

def get_gender_age_info(input_file, output_file):
    age_gender_list = []
    with open(input_file, "r", encoding='utf-8') as fhIn:
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                age = age_inference(line['includes']['users'][0]['description'])
                if age < 0:
                    age = age_inference(line['data']['text'])
                gender = gender_inference(line['includes']['users'][0]['description'])
                age_gender_list.append([line['includes']['users'][0]['id'], line['includes']['users'][0]['username'], gender, age])
            else:
                logging.warning(line)
                return
    df = pd.DataFrame(columns=['id', 'screen_name', 'gender', 'age'], data=age_gender_list)
    df.to_csv(output_file, index=False, encoding='utf-8')


if __name__ == "__main__":
    build_match_word_regex()
    get_sample_users(src_file_path, step1_output_file_name)
    get_gender_age_info(step1_output_file_name, step1_output_file_name)