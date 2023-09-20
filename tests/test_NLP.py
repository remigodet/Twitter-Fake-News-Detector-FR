def test_get_model():
    try :
        assert get_model() != print("No models")
        print('model defined successfully')
    except :
        print ('no model defined')

def test_credibility():
    try :
        assert credibility() is not None 
        print ('credibility defined successfully')
    except :
        print ('not sure enough to define a reliable credibility')
        
def test_get_data():
    try :
        assert get_data is not None 
        print ('Data succesfully downloaded')
    except :
        print('no data downloaded')