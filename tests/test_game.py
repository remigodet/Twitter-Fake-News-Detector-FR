def test_pick_tweet():
    try :
        (tweet, credibility)=pick_tweet()
        assert tweet is not None and (credibility is 0 or credibility is 1)
        print('the tweets are picked and the credibility correctly calculated')
    except : 
        print ('the tweet are not picked or th credibility is incorrect')

