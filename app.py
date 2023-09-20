import dash_html_components as html
import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output, State
from GAME.game_functions import pick_tweet
from data import load_datas


app = dash.Dash(__name__)

# test
nb_tweets = 5
score_prec = 0
b1_prec = 0
b2_prec = 0
hist = []
datas = load_datas('vaccin')
selec_prec = 0


# Cette partie sert à définir la forme de la page sous forme HTML

app.layout = html.Div(children=[
    html.Div(  # Partie de gauche
        children=[
            html.Button(id='reset', n_clicks=None,
                        className='but_reset', children='Rénitialiser'),
            html.Button(id='vaccins', children='Vaccins',
                        n_clicks=0, className='but_g'),
            html.Button(id='immigration', children='Immigration',
                        n_clicks=0, className='but_g'),
            html.Button(id='5g', children='5G',
                        n_clicks=0, className='but_g'),
            dcc.Input(id='recherche', type='text', className='recherche'),
            html.Button(id='but_rech', children='Rechercher',
                        n_clicks=0, className='but_rech')
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'width': '35%',
            'align-items': 'flex-start'
        }
    ),
    html.Div(  # Partie centrale
        children=[
            html.Header(  # En-tête
                children=[
                    html.H1("Ce tweet est-il vrai ?",
                            style={'font-family': 'chirp, Arial', 'text-align': 'center', 'color': '#1d9bf0'})  # Titre principal
                ]
            ),
            html.Div(  # Partie principale : le jeu
                children=[
                    html.Div(  # Tweet
                        children=[
                            html.P(id='tweet', style={
                                'font-family': 'Helvetica', 'font-size': '1.3em', 'text-align': 'center'})
                        ],
                        style={'border-width': '2px',
                               'border-color': '#cfd9de', 'border-style': 'solid', 'border-radius': '10px', 'width': '100%', 'margin': '25px', 'margin-left': '0px', 'margin-right': '0px'}
                    ),
                    html.Div(  # Boutons
                        children=[
                            html.Button(id='vrai', children='Vrai',
                                        n_clicks=0, style={'margin': '10px', 'font-size': '1.2em'}, className='but_reset'),
                            html.Button(id='faux', children='Faux', n_clicks=0, style={
                                        'margin': '10px', 'font-size': '1.2em'}, className='but_reset')

                        ],
                        style={'display': 'flex',
                               'justify-content': 'center'}
                    )
                ],
            ),
            html.Div(  # En bas, on te dit si t'as eu juste ou faux.
                children=[
                    html.P(id="Résultat")
                ]
            ),
            html.Div(id='histo', style={'margin-top': '70px', 'border-style': 'solid',
                     'border-radius': '10px', 'border-width': '1px', 'border-color': '#cfd9de', 'padding': '10px', 'height': '300px', 'overflow': 'scroll', 'font-family': 'Helvetica Neue'})  # Historique
        ],
        style={'width': '30%'}
    ),
    html.Div(  # Partie de droite
        children=[
            html.Div(children=[
                html.Img(src=app.get_asset_url('lauriers.png'),
                         style={'width': '110px', 'height': '100px', 'text-align': 'center'})],
                     style={'display': 'flex',
                            'justify-content': 'center', 'margin-top': '10px'}
                     ),
            html.H2(id='score',
                    style={'text-align': 'center', 'font-family': 'Helvetica',
                           'position': 'relative', 'bottom': '87px', 'color': '#303030'}
                    )

        ],
        style={'width': '35%'},
    )
], style={'display': 'flex'})


# Cette partie permet l'interaction entre la page et l'utilisateur


@app.callback(
    output=[  # sorties
        Output('tweet', 'children'),  # affiche le tweet
        Output('Résultat', 'children'),  # affiche le résultat précédent
        Output('Résultat', 'style'),  # pour changer la couleur de la phrase
        Output('score', 'children'),  # affiche le score
        Output('histo', 'children'),  # affiche l'historique
        Output('vaccins', 'className'),
        Output('immigration', 'className'),
        Output('5g', 'className'),
        Output('recherche', 'className')

    ],
    inputs=[  # entrées
        Input('vrai', 'n_clicks'),  # bouton de choix vrai
        Input('faux', 'n_clicks'),  # bouton de choix faux
        Input('vaccins', 'n_clicks'),
        Input('immigration', 'n_clicks'),
        Input('5g', 'n_clicks'),
        Input('but_rech', 'n_clicks')
    ],
    state=[State('recherche', 'value')]
)
def update_jeu(bouton1, bouton2, b1, b2, b3, b4, strt):
    """
    Cett fonction prends l'entrée données par l'utilisateur et donne les sorties à afficher dans l'application.
    Cette fonction sert de base à l'application est n'est destiné à être appelée directement par l'utilisateur.

    Entrées :

    bouton1 : int
        Nombre de clicks sur le bouton tweet 1

    bouton2 : int
        Nombre de clicks sur le bouton tweet 2

    Sorties :

    tweet : str
        Donne le tweet à afficher sur la page pour évaluation

    res : str
        Donne le résultat de la dernière instance de jeu

    style : dict
        Donne le style CSS à donner au résultat, ce qui permet d'adapter la couleur
        ex : {'font-family':'Helvetica'}

    score : int
        Donne le score actuel du joueur

    histo : [html.P]
        Liste de paragraphes à écrire sur les histo

    """
    global score_prec, selec_prec
    global nb_tweets
    global b1_prec, b2_prec, hist, datas
    global datas
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    classes = ['but_g', 'but_g', 'but_g', 'recherche']
    if 'vaccins' in changed_id:
        datas = load_datas('vaccin')
        selec_prec = 0
    elif 'immigration' in changed_id:
        datas = load_datas('immigration')
        selec_prec = 1
    elif '5g' in changed_id:
        datas = load_datas('5G')
        selec_prec = 2
    elif 'but_rech' in changed_id and strt != None:
        datas = load_datas(strt)
        selec_prec = 3
    if selec_prec < 3:
        classes[selec_prec] = 'but_g_selec'
    else:
        classes[3] = 'recherche_selec'

    style = {'font-size': '1.3em',  # style de base à appliquer au résultat
             'font-family': 'Arial', 'text-align': 'center'}
    style_histo = {'font-size': '1.3em',  # style de base à appliquer au résultat
                   'font-family': 'Arial'}
    # Détermination de quel bouton a été préssé (ou aucun)
    if bouton1 != b1_prec:  # bouton vrai
        b_presse = 1
        b1_prec = bouton1
    elif bouton2 != b2_prec:  # bouton faux
        b_presse = 0
        b2_prec = bouton2
    else:  # pas de bouton préssé
        b_presse = None
    # Donne la réponse (bonne ou mauvaise)
    if b_presse != None:
        # Bonne réponse
        if pick_tweet(datas, b1_prec+b2_prec-1)[1] == b_presse:
            res = 'Bonne réponse !'
            style['color'] = '#4ce600'  # bonne réponse en vert
            style_histo['color'] = '#4ce600'
            score_prec += 1
        else:  # Mauvaise réponse
            res = 'Mauvaise réponse !'
            style['color'] = 'red'  # mauvaise réponse en rouge
            style_histo['color'] = 'red'
    else:  # Pas encore de réponse (premier tweet évalué)
        res = ''
        score_prec = 0
    # Actualisation de l'historique
    if b_presse != None:
        hist.append(
            html.P(pick_tweet(datas, b1_prec+b2_prec-1)[0], style=style_histo))
    # Donne le tweet suivant à évaluer
    if len(datas.tweet_textual_content)-1 > b1_prec+b2_prec:
        tweet = pick_tweet(datas, b1_prec+b2_prec)[0]
    else:
        tweet = "Vous avez fini cette série"
        b1_prec, b2_prec, score_prec = 0, 0, 0
    return [tweet, res, style, str(score_prec)+'/'+str(b1_prec+b2_prec), hist[::-1]]+classes

# Bouton reset


@app.callback(
    output=[Output('vrai', 'n_clicks'), Output('faux', 'n_clicks')],
    inputs=[Input('reset', 'n_clicks')]
)
def reset(reset):
    """
    Cette fonction qui s'active quand le bouton rénitialiser est pressé a pour but de rénitialiser tous les données

    Entrées : 

    reset : int
        Nombre de clicks faits par le bouton rénitialiser

    Sorties : 

    Renvoie systématiquement [0,0] pour rénitialiser les boutons vrai et faux
    """
    global b1_prec, b2_prec, hist, score_prec
    b1_prec, b2_prec, hist, score_prec = 0, 0, [], 0
    return [0, 0]


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True)
