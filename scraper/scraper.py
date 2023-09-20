import tweepy


def collect(query, n, API, user=False):
    """
    Scrape for tweets based on queries
    :param query: (string) the text query to search on twitter
    :return: json of tweets
    """

    # return API.search_tweets(query, lang="fr", count=n)
    if user:
        tweets = API.user_timeline(
            screen_name=query, tweet_mode='extended', count=n)
    else:
        tweets = tweepy.Cursor(API.search_tweets,
                               q=query,
                               lang="fr",
                               tweet_mode='extended'
                               ).items(n)
    return tweets


if __name__ == '__main__':
    collect("helloworld")
