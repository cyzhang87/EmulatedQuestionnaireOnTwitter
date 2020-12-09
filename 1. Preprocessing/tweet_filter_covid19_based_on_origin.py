import ast
import logging
import re, pickle, os
import pandas as pd

logging.basicConfig(level=logging.INFO, filename="log-filter.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter.txt ...")


src_file_path = "C://cyzhang//project//m3inference-cyzhang//m3inference//twitter_cache//origin_filter_0807//"
dst_file_path = "C://cyzhang//project//m3inference-cyzhang//m3inference//twitter_cache//origin_filter_0807//"
keyword_file = "twitter_covid19_keywords_all.txt"
covid_dictionary_file = "twitter_covid19_keywords_all.csv"
word_pattern_file = "covid_word_pattern"

count = 0

ORIG_TWEET_FILE = src_file_path + 'origin_tweets.csv'
COVID_TWEET_FILE = dst_file_path + 'covid_tweets.csv'
tweets_label_file = src_file_path + 'tweets_covid_labels.csv'


keywords_list = []

"""
for word in open(keyword_file,'r'):
    keywords_list.append(word[:word.find('\n')].lower()) #remove '\n'

def has_keywords_in_content(content):
    for word in keywords_list:
        if word in content:
            return True
    return False
"""

if not os.path.exists(dst_file_path):
    os.mkdir(dst_file_path)

#spacy
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

keywords_patterns = []
pattern_template = {"label": "covid", "pattern": "covid-19"}
#nlp = spacy.load("en_core_web_sm")
nlp = English()
ruler = EntityRuler(nlp)
for word in open(keyword_file, 'r', encoding='utf-8'):
    pos = word.find('\n')
    if pos == -1:
        pos = len(word)
    keyword = word[:pos].lower()
    pattern_template = {"label": "covid"}
    pattern_template["pattern"] = keyword
    keywords_patterns.append(pattern_template)

ruler.add_patterns(keywords_patterns)
nlp.add_pipe(ruler)

def has_covid_pattern_in_content(content):
    doc = nlp(content)
    if len(doc.ents):
        return True
    return False
    #print([(ent.text, ent.label_) for ent in doc.ents])

def covid_filter():
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        return

    if os.path.isfile(COVID_TWEET_FILE):
        print('COVID_TWEET_FILE already exits\n')
    else:
        print('Start filter covid from origin...\n')

        file_object = open(COVID_TWEET_FILE, 'a', encoding='utf8')

        total_count = 0
        covid_count = 0
        for line in open(os.path.join(src_file_path, ORIG_TWEET_FILE),'r', encoding='utf-8'):
            total_count += 1
            record = ast.literal_eval(line) #json.dumps
            if 'data' in record:
                #if 'lang' not in record['data'] or record['data']['lang'] != 'en':
                #    continue

                if 'text' in record['data']:
                    content = record['data']['text']
                    content = content.lower()
                    if has_covid_pattern_in_content(content):
                        covid_count += 1
                        file_object.write("{}".format(line))
                        continue

            if 'includes' in record and 'tweets' in record['includes']:
                for subrecord in record['includes']['tweets']:
                    if 'text' in subrecord:
                        content = subrecord['text'].lower()
                        if has_covid_pattern_in_content(content):
                            covid_count += 1
                            file_object.write("{}".format(line))
                            break

        print('{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(ORIG_TWEET_FILE, total_count, covid_count, covid_count / total_count))
        logging.info('{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(ORIG_TWEET_FILE, total_count, covid_count, covid_count / total_count))

        #程序结束前关闭文件指针
        if file_object != None:
            file_object.close()
        return

#build word regex pattern

def read_data_from_pickle(infile):
    with open(infile, 'rb') as fp:
        return pickle.load(fp)


def save_data_to_pickle(outfile, all_tweets):
    with open(outfile, 'wb') as fp:
        pickle.dump(all_tweets, fp)


def build_match_word_regex():
    dictionary_file = covid_dictionary_file

    if os.path.isfile(word_pattern_file):
        word_pattern = read_data_from_pickle(word_pattern_file)
        print('Loaded word_pattern from file\n')
    else:
        dict_df = pd.DataFrame()
        if os.path.isfile(dictionary_file):
            dict_df = pd.read_csv(dictionary_file, usecols=['name'], encoding="ISO-8859-1")
        else:
            print("dictionary file is not exit.")
            exit()
        print('Generate word_pattern ...\n')
        word_set = set()
        for index, word in dict_df.iterrows():
            if type(word['name']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', word['name'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if len(drug_name) >= 3:
                    word_set.add(r'\b' + drug_name + r'(?![\w-])')
                else:
                    print(word['name'])

        word_list = list(word_set)
        word_list.sort(key=lambda i: len(i), reverse=True)
        word_pattern = re.compile('|'.join(word_list), re.IGNORECASE)
        save_data_to_pickle(word_pattern_file, word_pattern)

    return word_pattern

def generate_covid_label():
    word_pattern_file = build_match_word_regex()
    total_count = 0
    covid_count = 0
    label_list = []
    for line in open(os.path.join(src_file_path, ORIG_TWEET_FILE), 'r', encoding='utf-8'):
        total_count += 1
        record = ast.literal_eval(line)  # json.dumps
        if 'data' in record:
            # if 'lang' not in record['data'] or record['data']['lang'] != 'en':
            #    continue
            if 'text' in record['data']:
                content = record['data']['text']
                tweets_tmp = re.sub(r'[\r\n]|(\w+:\/\/\S+)|(&amp)|[^a-z0-9 ]', ' ', content.lower()).strip()
                # 去掉多余空格
                tweets_tmp = re.sub(' +', ' ', tweets_tmp.strip())
                match = re.search(word_pattern_file, tweets_tmp)
                if match:
                    covid_count += 1
                    label_list.append([str(record['data']['id']), 1])
                else:
                    label_list.append([str(record['data']['id']), 0])

    print('{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(ORIG_TWEET_FILE, total_count, covid_count,
                                                                        covid_count / total_count))
    logging.info(
        '{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(ORIG_TWEET_FILE, total_count, covid_count,
                                                                      covid_count / total_count))
    data_df = pd.DataFrame(columns=['id', 'covid_label'], data=label_list)
    data_df.to_csv(tweets_label_file, index=False)

if __name__ == "__main__":
    covid_filter()
    generate_covid_label()
    print("filter end")
