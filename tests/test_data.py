def test_generate_query():
    try :
        query = generate_query()
        assert query is not None
        print('query generated successfully')
    except :
        print('could not generate query')
    
def test_data_handling():
    try :
        select_interessant_data = data_handling()
        assert select_interessant_data != data and select_interessant_data is not None 
        print('data categories selected successfully')
    except :
        print('could not selection categories')
 
def test_save_tweets ():
    try :
         assert save_tweets() is not None 
         print('saved successfully')
    except :
        print ('not saved')
        
def test_scrape():
    try :
        assert scrape() is not None 
        print('tweets with keywords found succesfully')
    except :
        print ('Not enough tweets with the keyword given')
        
def test_json_to_df():
    try:
        df = json_to_df()
        assert df is not None
        print ('converted succesfully')
    except :
        print ('could not convert')
        
def test_get_data():
    try :
        assert get_data() is not None 
        print('tweets with list of words find')
    except : print('no tweets with such words')

def test_collected_to_df():
    try :
        assert collected_to_df is not None 
        print('dataframe created successfully')
    except :
        print ('no datafrme created')
 
def test_load_datas():
    try :
         assert load_datas() is not None 
         print('return data scrapped or category')
    except :
        print('no data returned') 
        
