import argparse
import numpy as np
from matplotlib import pyplot as plt
from pymongo import MongoClient
from mpl_toolkits.basemap import Basemap
import cPickle as pickle
from lib import segment

# get script arguments
parser = argparse.ArgumentParser()

parser.add_argument('--db', dest="db", required=True)
parser.add_argument('--coll', dest='coll', required=True,
                    help="Flights data collection name")
parser.add_argument('--skip', dest="skip", default=0, type=int)
parser.add_argument('--phase', dest="phase", type=str)
args = parser.parse_args()

DB = args.db
COLL = args.coll
SKIP = args.skip
PHASE = args.phase

# Configuration for the database
HOST = "localhost"
PORT = 27017

mongo_client = MongoClient('localhost', 27017)
mcoll = mongo_client[DB][COLL]

if PHASE:
    res = mcoll.find({'phase': PHASE}).skip(SKIP)
else:
    res = mcoll.find().skip(SKIP)

for r in res:
    data = r['data']
    icao = r['icao']

    if len(data) == 0:
        continue

    # pickle.dump(data, open('temp.pkl', 'wb'))

    data = np.asarray(data)

    times = data[:, 0].astype(int)
    times = times - times[0]
    lats = data[:, 1]
    lons = data[:, 2]
    alts = data[:, 3].astype(int)
    spds = data[:, 4].astype(int)

    # pickle.dump(data, open('data/segment_test_1116.pkl', 'wb'))

    labels = segment.fuzzylabels(times, alts, spds)
    colormap = {'GND': 'gray', 'CL': 'green', 'CR': 'blue', 'DE': 'olive',
                'NA': 'red'}

    colors = [colormap[l] for l in labels]

    # setup mercator map projection.
    plt.subplot(131)
    m = Basemap(llcrnrlon=min(lons)-2, llcrnrlat=min(lats)-2,
                urcrnrlon=max(lons)+2, urcrnrlat=max(lats)+2,
                resolution='l', projection='merc')
    m.fillcontinents()
    # plot SIL as a fix point
    latSIL = 51.989884
    lonSIL = 4.375374
    m.plot(lonSIL, latSIL, latlon=True, marker='o', c='red', zorder=9)

    plt.subplot(131)
    m.scatter(lons, lats, latlon=True, marker='.', c=colors, lw=0, zorder=10)

    plt.subplot(132)
    plt.ylim([-100, 40000])
    plt.scatter(times, alts, marker='.', c=colors, lw=0)

    plt.subplot(133)
    plt.ylim([0, 600])
    plt.scatter(times, spds, marker='.', c=colors, lw=0)

    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
