import json
from collections import namedtuple

with open('dataigdb.json') as datalstm:  
    games = json.load(datalstm)

with open('IGDB/dataplayer_perspectives.json') as dataplayer_perspectives:  
    player_perspectives = json.load(dataplayer_perspectives)

with open('IGDB/datagame_modes.json') as datagame_modes:  
    game_modes = json.load(datagame_modes)

with open('IGDB/datathemes.json') as datathemes:  
    themes = json.load(datathemes)

with open('IGDB/datagenres.json') as datagenres:  
    genres = json.load(datagenres)

with open('IGDB/dataplatforms.json') as dataplatforms:  
    platforms = json.load(dataplatforms)

for g in games:

    pps = g['player_perspectives']
    g['player_perspectives'] = []
    for pp in pps:
        for dpp in player_perspectives:
            if dpp['id'] == pp:
                g['player_perspectives'].append(dpp['name'])

    gmd = g['game_modes']
    g['game_modes'] = []
    for gm in gmd:
        for dgm in game_modes:
            if dgm['id'] == gm:
                g['game_modes'].append(dgm['name'])

    tem = g['themes']
    g['themes'] = []
    for t in tem:
        for dt in themes:
            if dt['id'] == t:
                g['themes'].append(dt['name'])

    gen = g['genres']
    g['genres'] = []
    for gn in gen:
        for dgn in genres:
            if dgn['id'] == gn:
                g['genres'].append(dgn['name'])

    # pf = g['platforms']
    # g['platforms'] = []
    # for p in pf:
    #     for dp in platforms:
    #         if dp['id'] == p:
    #             g['platforms'].append(dp['name'])

    if (0 <= g['aggregated_rating'] <= 19):
        g['aggregated_rating'] = 'Dislike'
    elif (19 <= g['aggregated_rating'] <= 49):
        g['aggregated_rating'] = 'Unfavorable'
    elif (49 <= g['aggregated_rating'] <= 74):
        g['aggregated_rating'] = 'Average'
    elif (74 <= g['aggregated_rating'] <= 89):
        g['aggregated_rating'] = 'Favorable'
    elif (89 <= g['aggregated_rating'] <= 100):
        g['aggregated_rating'] = 'Acclaim'

filename = 'datasetlstm.json'
with open(filename, 'w+') as outfile:
    json.dump(games, outfile)