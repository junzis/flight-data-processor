import os
import json
import numpy as np

curr_path = os.path.dirname(os.path.realpath(__file__))
f_airports = curr_path + "/db/airports.json"
f_aircrafts = curr_path + "/db/aircrafts.json"


def get_closest_airport(lat, lon):
    keys = []
    coords = []
    with open(f_airports) as f:
        airports = json.load(f)
        for k, v in airports.items():
            keys.append(k)
            coords.append([v['lat'], v['lon']])
    coords = np.asarray(coords)
    dist2 = np.sum((coords - [lat, lon])**2, axis=1)
    idx = np.argmin(dist2)
    return airports[keys[idx]]


def get_aircraft(mdl):
    with open(f_aircrafts) as f:
        acs = json.load(f)

    if mdl in acs:
        return acs[mdl]
    else:
        return None
