import tweepy
from scraper.credentials import *


def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with an access keys provided in a file credentials.py
    :return: the authentified API
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)

    try:
        api.verify_credentials()
    except Exception as e:
        raise e
    return api

if __name__ == '__main__':
    test_twitter_setup()
