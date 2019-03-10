import json
from igdb_api_python.igdb import igdb

igdb = igdb("CHAVE_DA_IPA")

for i in range (0, 100):
    result = igdb.games({
        'filters' :{
            "[summary][exists]": '',
            "[storyline][exists]": '',
            "[genres][exists]": '',
            "[themes][exists]": '',
            "[game_modes][exists]": '',
            "[player_perspectives][exists]": '',
            "[aggregated_rating][exists]": ''
        },
        'limit': 50,
        'offset': i*50,
        'order':"date:asc",
        'fields': [ 'summary',
                    'storyline',
                    'genres',
                    'themes',
                    'game_modes',
                    'player_perspectives',
                    'aggregated_rating'
                ]
    })
    filename = 'datasetigdb.json'
    with open(filename, 'a') as outfile:
        json.dump(result.body, outfile)