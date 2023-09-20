import pandas as pd
import data as dt
datas = pd.DataFrame({'tweet_textual_content': ['La terre est plate', 'Le citron ne soigne pas du cancer',
                     'Les illuminatis sont des extraterrestres', "Se brosser les dents prévient l'analphabétisme", "Se laver prévient de la puanteur"]})



def pick_tweet(datas, num_round):
    '''
    input:  data: un dataframe panda contenant une colonne tweet_textual_content
            num_round: le numéro de la manche en cours (attention, on commence à zéro)
    output: un couple (string,int) contenant un tweet et sa crédibilité'''
    return datas.tweet_textual_content.iloc[num_round], datas.credibility.iloc[num_round]


if __name__ == '__main__':
    print(pick_tweet('immigration', 1))
