import sys
import argparse
import numpy as np
from time import time
from itertools import cycle
from matplotlib import pyplot as plt
from sklearn import preprocessing
from pymongo import MongoClient
from sklearn.cluster import Birch, MeanShift, DBSCAN

# Constants
HOST = "localhost"   # MongoDB host
PORT = 27017         # MongoDB port
MIN_DATA_SIZA = 100  # minimal number of data in a flight
CHUNK_SIZE = 50      # number of icaos to be processed in chunks
TEST_FLAG = True     # weather this is a test run

# get script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db', dest="db", required=True)
parser.add_argument('--inColl', dest='input_coll', required=True,
                    help="Postion collection name")
parser.add_argument('--outColl', dest='output_coll',
                    help="Flight collection name")
args = parser.parse_args()

mdb = args.db
pos_coll = args.input_coll
flights_coll = args.output_coll

if flights_coll == pos_coll:
    sys.exit("Error: Output and input collections can not be the same!")

mongo_client = MongoClient('localhost', 27017)
mcollpos = mongo_client[mdb][pos_coll]

if flights_coll:
    TEST_FLAG = False
    mcollflights = mongo_client[mdb][flights_coll]
    mcollflights.drop()    # clear the segment collection first

print "[1] Querying database."

# Find all ICAO IDs in the dataset
stats = mcollpos.aggregate([
    {
        '$group': {
            '_id': '$icao',
            'count': {'$sum': 1}
        }
    }
])

icaos = []
for ac in stats['result']:
    if ac['count'] > MIN_DATA_SIZA:
        icaos.append(ac['_id'])

print "[2] %d number of valid ICAOs." % len(icaos)

for i in xrange(0, len(icaos), CHUNK_SIZE):

    print '[3][%d-%d of %d] ICAOs beening processed.' \
            % (i, i+CHUNK_SIZE, len(icaos))

    chunk = icaos[i: i+CHUNK_SIZE]

    ids = []
    lats = []
    lons = []
    alts = []
    spds = []
    hdgs = []
    rocs = []
    times = []

    print "  [a] fetching records"
    allpos = mcollpos.find({'icao': {'$in': chunk}})

    for pos in allpos:
        if 'spd' not in pos:
            pos['spd'] = np.nan

        if 'hdg' not in pos:
            pos['hdg'] = np.nan

        ids.append(pos['icao'])
        lats.append(pos['loc']['lat'])
        lons.append(pos['loc']['lng'])
        alts.append(float(pos['alt']))
        spds.append(pos['spd'])
        hdgs.append(pos['hdg'])
        rocs.append(pos['roc'])
        times.append(float(pos['ts']))

    times = np.array(times)
    alts = np.array(alts)

    #####################################################################
    # Continous fligh path extraction using machine learning algorithms
    #####################################################################

    print "  [b] data scaling"

    # transform the text ids into numbers
    # ------------------------------------
    le = preprocessing.LabelEncoder()
    encoded_ids = le.fit_transform(ids)

    # Apply feature scaling - on altitude, spds, and times
    # ------------------------------------------------------
    mms = preprocessing.MinMaxScaler(feature_range=(0, 1000))
    times_norm = mms.fit_transform(times.reshape((-1, 1)))
    dt = mms.scale_ * 0.5 * 60 * 60   # time interval of 30 mins

    mms = preprocessing.MinMaxScaler(feature_range=(0, 100))
    alts_norm = mms.fit_transform(alts.reshape((-1, 1)))

    print "  [c] creating machine learning dataset."

    # aggregate data by there ids
    # ----------------------------
    acs = {}
    for i in xrange(len(ids)):
        if ids[i] not in acs.keys():
            acs[ids[i]] = []

        acs[ids[i]].append([times_norm[i], alts_norm[i], int(times[i]),
                            lats[i], lons[i], int(alts[i]),
                            spds[i], hdgs[i], rocs[i]])

    print "  [d] start clustering, and saving results to DB"

    # Apply clustering method
    # ------------------------
    cluster = DBSCAN(eps=dt, min_samples=MIN_DATA_SIZA)
    # cluster = Birch(branching_factor=50, n_clusters=None, threshold=10)

    acsegs = {}
    total = len(acs.keys())
    count = 0
    print '    * processing ',
    for k in acs.keys():
        count += 1
        print '.',

        data = np.asarray(acs[k])

        tdata = np.copy(data)
        tdata[:, 1:] = 0
        cluster.fit(tdata)
        labels = cluster.labels_
        n_clusters = np.unique(labels).size
        # print("n_clusters : %d" % n_clusters)

        if not TEST_FLAG:
            for i in range(n_clusters):
                mask = labels == i
                c = data[mask, 2:]
                c = c[c[:, 0].argsort()]   # sort by ts
                if len(c) > MIN_DATA_SIZA:
                    mcollflights.save({'icao': k,
                                       'data': c.tolist()})

        # Plot result
        if TEST_FLAG:
            colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
            for i, c in zip(range(n_clusters), colorset):
                mask = labels == i
                ts = data[mask, 0].tolist()
                alts = data[mask, 5].tolist()
                if len(alts) > MIN_DATA_SIZA:
                    plt.plot(ts, alts, 'w', color=c, marker='.', alpha=1.0)

            plt.xlim([0, 1000])
            plt.draw()
            plt.waitforbuttonpress(-1)
            plt.clf()
    print ''

print
print "[4] All completed"
