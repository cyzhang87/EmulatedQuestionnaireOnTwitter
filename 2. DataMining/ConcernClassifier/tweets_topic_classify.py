import pandas as pd
import re, pickle, os
import logging
import ast

logging.basicConfig(level=logging.INFO, filename="log.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log.txt ...")

test_switch = False
min_valid_word_len = 3

dir_name = "two_filter_0807"
tweets_file = "../twitter_cache/" + dir_name + "/twitter_sample-20200807135344_en.csv"
tweets_label_file = "../data/" + dir_name + "/tweets_labels.csv"

pattern_path = "../data/patterns/"
input_path = "../data/patterns_generation_files/"

chemical_file = input_path + "DrugDictionary.csv"
economic_file = input_path + "EconomicsDictionary.csv"
stop_word_file = input_path + "SmartStoplist.txt"
not_drug_word_file = input_path + "NotDrugWord.txt"
new_drug_word_file = input_path + "NewDrugWord.txt"
drug1_file = input_path + "DrugDictionary_labeled.csv"
drug2_file = input_path + "NewDrugWord_labeled.csv"
politics_file = input_path + "PoliticsDictionary.csv"
education_file = input_path + "EducationDictionary.csv"
entertainment_file = input_path + "EntertainmentDictionary.csv"


chemical_word_pattern_file = pattern_path + "chemical_word_pattern"
drug_word_pattern_file = pattern_path + "drug_word_pattern"
economic_word_pattern_file = pattern_path + "economic_word_pattern"
politics_word_pattern_file = pattern_path + "politics_word_pattern"
education_word_pattern_file = pattern_path + "education_word_pattern"
entertainment_word_pattern_file = pattern_path + "entertainment_word_pattern"

if not os.path.exists(os.path.dirname(tweets_label_file)):
    os.mkdir(os.path.dirname(tweets_label_file))
	
def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                word = re.sub(r'[^a-z0-9 ]', ' ', word.lower()).strip()
                word = re.sub(' +', ' ', word.strip())
                stop_words.append(word)
    return stop_words


def load_words_file(file_name):
    words = []
    for line in open(file_name):
        for word in line.split('\n'):
            if word != '':
                word = re.sub(r'[^a-z0-9 ]', ' ', word.lower()).strip()
                word = re.sub(' +', ' ', word.strip())
                words.append(word)
    return words


def read_data_from_pickle(infile):
    with open(infile, 'rb') as fp:
        return pickle.load(fp)


def save_data_to_pickle(outfile, all_tweets):
    with open(outfile, 'wb') as fp:
        pickle.dump(all_tweets, fp)


def build_chemicals_word_regex():
    if os.path.isfile(chemical_word_pattern_file):
        # Read cleaned tweets from saved file
        chemicals_word_pattern = read_data_from_pickle(chemical_word_pattern_file)
        print('Loaded chemicals_word_pattern from file\n')
    else:
        chemicals_df = pd.DataFrame()
        if os.path.isfile(chemical_file):
            chemicals_df = pd.read_csv(chemical_file, usecols=['PROPRIETARYNAME', 'NONPROPRIETARYNAME'],
                                       sep="\t")  # usecols=['ChemicalName', 'Synonyms'], encoding="ISO-8859-1"
        else:
            print("chemical file is not exit.")
            exit()
        print('Generate chemicals_word_pattern ...\n')
        stop_word_list = set(load_stop_words(stop_word_file)) #+ load_words_file(not_drug_word_file))
        drug_set = set()
        for index, chemical in chemicals_df.iterrows():
            # chemicalname中可能会有正则表达式特殊字符，把所有特殊字符替换成空格，并去掉前后空格
            if type(chemical['PROPRIETARYNAME']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', chemical['PROPRIETARYNAME'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if drug_name not in stop_word_list:
                    if len(drug_name) > min_valid_word_len:
                        drug_set.add(r'\b' + drug_name + r'(?![\w-])')  # r'\b' + chemical['ChemicalName'] + r'(?![\w-])'
            if type(chemical['NONPROPRIETARYNAME']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', chemical['NONPROPRIETARYNAME'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if drug_name not in stop_word_list:
                    if len(drug_name) > min_valid_word_len:
                        drug_set.add(r'\b' + drug_name + r'(?![\w-])')

        for line in open(new_drug_word_file):
            for word in line.split('\n'):
                if word != '':
                    word = re.sub(r'[^a-z0-9 ]', ' ', word.lower()).strip()
                    word = re.sub(' +', ' ', word.strip())
                    drug_set.add(r'\b' + word + r'(?![\w-])')

        drug_list = list(drug_set)
        drug_list.sort(key=lambda i: len(i), reverse=True)
        chemicals_word_pattern = re.compile('|'.join(drug_list), re.IGNORECASE)
        save_data_to_pickle(chemical_word_pattern_file, chemicals_word_pattern)

    return chemicals_word_pattern


def build_three_classes_chemicals_word_regex():
    if os.path.isfile(drug_word_pattern_file):
        drug_patterns = read_data_from_pickle(drug_word_pattern_file)
        print('Loaded drug_word_pattern from file\n')
    else:
        if os.path.isfile(drug1_file):
            drug1_df = pd.read_csv(drug1_file, usecols=['PROPRIETARYNAME', 'NONPROPRIETARYNAME', 'LABEL'],
                                   sep="\t")  # usecols=['ChemicalName', 'Synonyms'], encoding="ISO-8859-1"
            drug2_df = pd.read_csv(drug2_file, usecols=['drug_name', 'label'])
        else:
            print("drug dictionary file is not exit.")
            exit()
        print('Generate drug_word_pattern ...\n')
        stop_word_list = set(load_stop_words(stop_word_file)) # + load_words_file(not_drug_word_file))
        drug_classes_dict = dict()
        drug_classes_dict['covid19_prophylactic_drug'] = set()
        drug_classes_dict['covid19_therapeutic_drug'] = set()
        drug_classes_dict['other_drug'] = set()

        for index, chemical in drug1_df.iterrows():
            if type(chemical['PROPRIETARYNAME']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', chemical['PROPRIETARYNAME'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if drug_name not in stop_word_list:
                    if len(drug_name) > min_valid_word_len:
                        drug_classes_dict[chemical['LABEL']].add(r'\b' + drug_name + r'(?![\w-])')
            if type(chemical['NONPROPRIETARYNAME']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', chemical['NONPROPRIETARYNAME'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if drug_name not in stop_word_list:
                    if len(drug_name) > min_valid_word_len:
                        drug_classes_dict[chemical['LABEL']].add(r'\b' + drug_name + r'(?![\w-])')

        for index, chemical in drug2_df.iterrows():
            drug_name = re.sub(r'[^a-z0-9 ]', ' ', chemical['drug_name'].lower()).strip()
            drug_name = re.sub(' +', ' ', drug_name.strip())
            if len(drug_name) > min_valid_word_len:
                    drug_classes_dict[chemical['label']].add(r'\b' + drug_name + r'(?![\w-])')

        drug_patterns = []
        drug_list = list(drug_classes_dict['covid19_prophylactic_drug'])
        drug_list.sort(key=lambda i: len(i), reverse=True)
        drug_patterns.append(re.compile('|'.join(drug_list), re.IGNORECASE))
        drug_list = list(drug_classes_dict['covid19_therapeutic_drug'])
        drug_list.sort(key=lambda i: len(i), reverse=True)
        drug_patterns.append(re.compile('|'.join(drug_list), re.IGNORECASE))
        drug_list = list(drug_classes_dict['other_drug'])
        drug_list.sort(key=lambda i: len(i), reverse=True)
        drug_patterns.append(re.compile('|'.join(drug_list), re.IGNORECASE))

        save_data_to_pickle(drug_word_pattern_file, drug_patterns)

    return drug_patterns

if test_switch:
    text = "Compatibility (0.017ferrocene)amylose of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    text = re.sub(r'(&amp)|[^a-z0-9 ]', ' ', text.lower()).strip()
    chemicals_word_pattern = build_chemicals_word_regex()
    if re.search(chemicals_word_pattern, text):
        print("text includes drug")
    else:
        print("text not includes drug")

def build_match_word_regex(selection):
    if selection == "3":  # economic
        word_pattern_file = economic_word_pattern_file
        dictionary_file = economic_file
    elif selection == "4": #politics
        word_pattern_file = politics_word_pattern_file
        dictionary_file = politics_file
    elif selection == "5": #education
        word_pattern_file = education_word_pattern_file
        dictionary_file = education_file
    elif selection == "6": #entertainment
        word_pattern_file = entertainment_word_pattern_file
        dictionary_file = entertainment_file


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
        stop_word_list = set(load_stop_words(stop_word_file))
        for index, word in dict_df.iterrows():
            if type(word['name']) == type('a'):
                drug_name = re.sub(r'[^a-z0-9 ]', ' ', word['name'].lower()).strip()
                drug_name = re.sub(' +', ' ', drug_name.strip())
                if drug_name not in stop_word_list:
                    if len(drug_name) > min_valid_word_len:
                        word_set.add(r'\b' + drug_name + r'(?![\w-])')

        word_list = list(word_set)
        word_list.sort(key=lambda i: len(i), reverse=True)
        word_pattern = re.compile('|'.join(word_list), re.IGNORECASE)
        save_data_to_pickle(word_pattern_file, word_pattern)

    return word_pattern


def generate_tweet_labels():
    chemicals_word_pattern = build_chemicals_word_regex()
    drug_patterns = build_three_classes_chemicals_word_regex()

    # economic, politic ... pattern
    word_pattern = []
    for type in range(3, 7):
        word_pattern.append(build_match_word_regex(str(type)))

    label_list = []
    with open(tweets_file, "r", encoding='utf-8') as fhIn:
        count = 0
        for line in fhIn:
            line = ast.literal_eval(line)  # to dict
            tweets_tmp = re.sub(r'[\r\n]|(\w+:\/\/\S+)|(&amp)|[^a-z0-9 ]', ' ', line['data']['text'].lower()).strip()
            # 去掉多余空格
            tweets_tmp = re.sub(' +', ' ', tweets_tmp.strip())

            # 1. drug and drug class label
            match = re.search(chemicals_word_pattern, tweets_tmp)
            if match:
                drug_label = 1
                drug_class_label = 2
                for i in range(3):
                    match = re.search(drug_patterns[i], tweets_tmp)
                    if match:
                        drug_class_label = i
                        break
            else:
                drug_label = 0
                drug_class_label = 3

            # 2. economic, politic ... label
            labels = []
            for i in range(4):
                match = re.search(word_pattern[i], tweets_tmp)
                if match:
                    label = 1
                else:
                    label = 0
                labels.append(label)

            label_list.append([str(line["data"]["id"]), drug_label, drug_class_label, labels[0], labels[1], labels[2], labels[3]])
            count += 1

            if count % 5000 == 0:
                print(count)

        data_df = pd.DataFrame(columns=['id', 'drug_label', 'drug_class_label', 'eco_label', 'pol_label', 'edu_label', 'ent_label'], data=label_list)
        data_df.to_csv(tweets_label_file, index=False)

if __name__ == "__main__":
    generate_tweet_labels()
