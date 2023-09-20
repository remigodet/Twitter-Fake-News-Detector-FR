from scraper.scraper import collect
from scraper.twitter_connection_setup import twitter_setup
import pandas as pd
from pandas import json_normalize
from os import listdir
import json
import codecs
import NLP.nlp as nlp


def generate_query(keywords):
    query = " ".join(keywords)
    return query


def data_handling(data):
    # keep only important features + ordering as in data_10_11_2020.json id  !!!
    '''
    :param json file: orginal json with too much data
    :return json file: usable json file
    '''

    data = json_normalize(data, max_level=1)
    categories = {"tweet_textual_content": {},
                  "ID": {}, "Date": {},
                  "Source": {}, "Likes": {}, "RTs": {}, "lang": {}}

    for i in range(len(data)):
        categories["tweet_textual_content"]["{}".format(
            i)] = data.iloc[i]["full_text"]
        categories["ID"]["{}".format(i)] = data.iloc[i]["id"]
        categories["Date"]["{}".format(i)] = data.iloc[i]["created_at"]
        categories["Source"]["{}".format(i)] = data.iloc[i]["source"]
        categories["Likes"]["{}".format(i)] = data.iloc[i]["favorite_count"]
        categories["RTs"]["{}".format(i)] = data.iloc[i]["retweet_count"]
        categories["lang"]["{}".format(i)] = data.iloc[i]["lang"]
    categories = pd.DataFrame(categories)
    # print(json.dumps(categories))
    return categories.to_dict()


def save_tweets(query, data):
    tweets = []
    for item in data:
        tweets.append(item._json)
    with open('Data/{}.json'.format(query), 'w', encoding='utf-8') as f:
        json.dump(data_handling(tweets), f)


def scrape(keywords, n, user=False):
    '''
    :param keywords: (list) words to search twitter
    :param n: (int) number of tweets scraped
    # TODO other params ?
    :return data:  pandas dataframe of scraped tweet containing the keywords
    '''
    # TODO: keywords from dash webpage
    try:
        API = twitter_setup()
    except:
        print("api connection failed")
    # Ne scrape que si pas déjà scrapé
    if type(get_data(keywords[0]+'.json')) == type(None):
        query = generate_query(keywords)
        if user:
            search_results = collect(query, n, API, user=user)
        else:
            search_results = collect(query, n, API)
        save_tweets(query, search_results)


def json_to_df(file):
    '''
    :param file: tweet file
    :return data: dataframe
    '''
    return pd.read_json('Data/{}'.format(file))


def get_data(file, words=[]):
    '''
    :param file: the file where is data writen like 'query.json'
    :param words: filter data with keywords
    Returns filtered data
    '''
    # TODO 2 file agglomerate with query + lister les dossier possibles dans dash
    # TODO 3 filter words from bash
    stored_files = listdir("./Data")

    if file == None:
        print("no file specified, try with example.json")
        return None
    elif file not in stored_files:
        print("file not found 404")
        return None
    else:
        data = json_to_df(file)

    if words == []:
        return data
    else:
        return data[any(word in data['tweet_textual_content']
                        for word in words)]


def collected_to_df():
    '''
    this function take every labeled tweets in files "fake" and "true" and create
    a dataframe (tweet_textual_content,credibility).
    '''
    fake_files = listdir("NLP/labeled_data/fake")
    fake_files.remove('echo')
    print(fake_files)
    data = pd.DataFrame({'tweet_textual_content': [], 'credibility': []})
    for name in fake_files:
        ff = codecs.open('NLP/labeled_data/fake/'+name, 'r', 'utf-8')
        tweet = ff.readline()
        data.loc[len(data)] = [tweet, 0]
        ff.close()
    true_files = fake_files = listdir("NLP/labeled_data/true")
    true_files.remove('echo')
    print(true_files)
    for name in true_files:
        ft = codecs.open('NLP/labeled_data/true/'+name, 'r', 'utf-8')
        tweet = ft.readline()
        data.loc[len(data)] = [tweet, 1]
        ft.close()
    return(data)


def load_datas(key):
    '''
    input: a keyword
    Output: the datas (tweet_textual_content,credibility) as a dataframe
    '''
    if key == 'vaccin':
        datas = pd.read_csv('Data/prepared_data/vaccin.csv')
        datas.sample(frac=1)
        return datas
    elif key == '5G':
        datas = pd.read_csv('Data/prepared_data/5G.csv')
        return datas.sample(frac=1)
    elif key == 'immigration':
        datas = pd.read_csv('Data/prepared_data/immigration.csv')
        return datas.sample(frac=1)
    else:
        scrape([key+' -RT'], 120)
        L = get_data(key+' -RT.json')['tweet_textual_content']
        dt = pd.DataFrame({'tweet_textual_content': L, 'credibility': [
                          nlp.credibility(tweet) for tweet in L]})
        col = pd.Series(map(lambda x: 0 if x < 0.45 else (
            1 if x > 0.6 else None), dt['credibility']))
        dt['credibility'] = col
        dt.dropna(inplace=True)
        return dt


if __name__ == '__main__':
    # assert generate_query(
    # ["emmanuael", "macron", "#presidentielle"]) == "emmanuael macron #presidentielle"

    # scrape(["zevent -RT"], 200)
    # print(listdir("./Data/tweets"))
    # print(get_data("example.json"))
    # with open("Data/tweets/data.json") as file:
    # print(data_handling(file))
    # print(collected_to_df())
    print(load_datas('5G'))
