import os
import requests
from requests import exceptions
from requests.auth import AuthBase
import json
import time
from m3inference.m3twitter import get_extension
from m3inference.preprocess import download_resize_img
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')

user_info_file = ""
user_image_dir = ""
new_user_info_file = ""
userremain_file = ""
#App: xxx_test_app
consumer_key_0 = "XXXX"  # Add your API key here
consumer_secret_0 = "XXXX"  #Add your API secret here
user_notfound_list = []

# Gets a bearer token
class BearerTokenAuth(AuthBase):
    def __init__(self, consumer_key, consumer_secret):
        self.bearer_token_url = "https://api.twitter.com/oauth2/token"
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        try:
            response = requests.post(
                self.bearer_token_url,
                auth=(self.consumer_key, self.consumer_secret),
                data={'grant_type': 'client_credentials'},
                headers={"User-Agent": "TwitterDevSampledStreamQuickStartPython"},
                proxies=proxies)

            if response.status_code is not 200:
                raise Exception(f"Cannot get a Bearer token (HTTP %d): %s" % (response.status_code, response.text))

            body = response.json()
            return body['access_token']

        except requests.exceptions.ConnectionError:
            logging.warning('ConnectionError')
        except:
            logging.warning('Unfortunitely -- An Unknow Error Happened')

    def __call__(self, r):
        r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
        return r

def get_profile_by_id(count_offset, lineno_list, userid_list, line_json_list, auth):
    global user_notfound_list
    id_str = ','.join(userid_list)
    url = "https://api.twitter.com/1.1/users/lookup.json?"
    params = {"user_id": id_str}
    try:
        response = requests.get(url, auth=auth, params=params, proxies=proxies)
        if response.status_code is not 200:
            logging.warning(str(response.status_code) + response.text)
            return False

        count = 0
        for user in json.loads(response.text):
            while userid_list[count] != user['id_str'] and count < len(userid_list):
                user_notfound_list.append(userid_list[count])
                logging.warning("userid: {} not found".format(userid_list[count]))
                count += 1
            count += 1
            img_path = user["profile_image_url_https"]
            img_path = img_path.replace("_normal", "_400x400")
            dotpos = img_path.rfind(".")
            # for some img_path, there is no '.' and extention, eg. https://pbs.twimg.com/profile_images/1303391057/XW0BrwDh_normal
            dotpos_pic = img_path.rfind("_400x400")
            if dotpos_pic > dotpos:
                img_file_resize = "{}/{}_224x224.jpg".format(user_image_dir, user["id"])
            else:
                img_file_resize = "{}/{}_224x224.{}".format(user_image_dir, user["id"], get_extension(img_path))
            if not os.path.isfile(img_file_resize):
                try_count = 5
                ret = download_resize_img(img_path, img_file_resize)
                while ret < -1 and try_count > 0:
                    logging.warning("userid: {}, ret: {}, try_count: {}".format(user["id"], ret, try_count))
                    ret = download_resize_img(img_path, img_file_resize)
                    try_count -= 1

                # can not find the pic ,so use the default
                if ret <= -1:
                    logging.warning(
                        "userid: {} has not download picture, use default_profile_400x400.png instead. ret: {}, try_count: {}. ".format(
                            user["id"], ret, try_count))
                    img_file_resize = os.path.join(os.path.dirname(img_file_resize), 'default_profile_400x400.png')

            # update json
            line_json_list[lineno_list[count-1]-count_offset]["img_path"] = img_file_resize

        with open(new_user_info_file, 'a') as outfile:
            for j in line_json_list:
                outfile.write(json.dumps(j) + '\n')

        return True
    except exceptions.Timeout as e:
        logging.warning('Timeout: ' + str(e))
    except exceptions.HTTPError as e:
        logging.warning('HTTPError: ' + str(e))
    except requests.exceptions.ConnectionError as e:
        logging.warning('ConnectionError: ' + str(e))
    except requests.exceptions.ChunkedEncodingError as e:
        logging.warning('ChunkedEncodingError: ' + str(e))
    except:
        logging.warning('Unfortunitely, an unknow error happened, please wait 3 seconds')
    return False

def update_profile(count_offset, lineno_list, userid_list, line_json_list, auth):
    status = get_profile_by_id(count_offset, lineno_list, userid_list, line_json_list, auth=auth)
    while status != True:
        logging.warning("get_profile_by_id error")
        time.sleep(3)
        status = get_profile_by_id(count_offset, lineno_list, userid_list, line_json_list, auth=auth)

def get_all_profiles(auth):
    line_count = 0
    id_count = 0
    lineno_list = []
    userid_list = []
    line_json_list = []
    with open(user_info_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line_json = json.loads(line)
            line_json_list.append(line_json)
            if 'default_profile_400x400.png' in line_json['img_path']:
                lineno_list.append(line_count)
                userid_list.append(line_json['id'])
                id_count += 1
            line_count += 1

            if id_count == 100:
                update_profile(line_count, lineno_list, userid_list, line_json_list, auth=auth)
                id_count = 0
                lineno_list = []
                userid_list = []
                line_json_list = []
                logging.info("line no: {}".format(line_count))

        if id_count > 0:
            update_profile(line_count, lineno_list, userid_list, line_json_list, auth=auth)
            logging.info("line no: {}".format(line_count))

    if not os.path.isfile(new_user_info_file):
        with open(new_user_info_file, 'a') as outfile:
            for j in line_json_list:
                outfile.write(json.dumps(j) + '\n')
    file_object = open(userremain_file, 'a', encoding='utf8')
    userremain_str = '\n'.join(user_notfound_list)
    file_object.write(userremain_str)
    file_object.close()


def update_new_profile_img(src_file, output_img_dir, dest_file, output_unfound_usrfile):
    global user_info_file, user_image_dir, new_user_info_file, userremain_file
    bearer_token = BearerTokenAuth(consumer_key_0, consumer_secret_0)

    if bearer_token.bearer_token == None :
        logging.warning("error -- has no bearer_token")
        return

    user_info_file = src_file
    user_image_dir = output_img_dir
    new_user_info_file = dest_file
    userremain_file = output_unfound_usrfile

    get_all_profiles(bearer_token)

if __name__ == "__main__":
    update_new_profile_img(user_info_file, user_image_dir, new_user_info_file, userremain_file)

