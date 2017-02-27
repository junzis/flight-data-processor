import argparse
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from pymongo import MongoClient
from mpl_toolkits.basemap import Basemap
from lib import segment

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B


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

plt.figure(figsize=(18, 6))

for r in res:
    icao = r['icao']

    times = np.array(r['ts']).astype(int)
    times = times - times[0]
    lats = np.array(r['lat'])
    lons = np.array(r['lon'])
    alts = np.array(r['alt'])

    spds = np.array([int(i) if len(str(i))>0 else np.nan for i in r['spd']])
    rocs = np.array([int(i) if len(str(i))>0 else np.nan for i in r['roc']])

    spds = fill_nan(spds)
    rocs = fill_nan(rocs)

    try:
        labels = segment.fuzzylabels(times, alts, spds, rocs)
    except:
        continue

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
    plt.ylim([-100, 45000])
    plt.scatter(times, alts, marker='.', c=colors, lw=0)

    plt.subplot(133)
    plt.ylim([0, 600])
    plt.scatter(times, spds, marker='.', c=colors, lw=0)

    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
