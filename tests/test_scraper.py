def test_twitter_connection_setup():
    try:
        api = twitter_setup()
        assert api is not None
        print('API active')
    except:
        print('Could not solve api connection')
        
def test_collect():
    try :
        tweets=collect()
        assert tweets is not None 
        print('tweet collected')
    except :
        print('could not collect tweets')
        