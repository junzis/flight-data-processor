import argparse
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import DBSCAN
from lib import segment

# get script arguments
parser = argparse.ArgumentParser()

parser.add_argument('--db', dest="db", required=True)
parser.add_argument('--inColl', dest='coll_in', required=True)
parser.add_argument('--outColl', dest='coll_out', required=True)
args = parser.parse_args()

DB = args.db
COLL_IN = args.coll_in
COLL_OUT = args.coll_out

# Configuration for the database
HOST = "localhost"
PORT = 27017

mongo_client = MongoClient('localhost', 27017)
mcollin = mongo_client[DB][COLL_IN]
mcollout = mongo_client[DB][COLL_OUT]

# clear output database
mcollout.drop()

res = mcollin.find()
total = res.count()

# define a cluster for extract sections within segments
dt = 180     # seconda
MIN_DATA_SIZA = 30  # number of samples
cluster = DBSCAN(eps=dt, min_samples=MIN_DATA_SIZA)

count = 0
for r in res:
    count += 1
    data = r['data']
    icao = r['icao']

    if len(data) == 0:
        continue

    data = np.array(data)

    times = data[:, 0].astype(int)
    alts = data[:, 3].astype(int)
    spds = data[:, 4].astype(int)

    labels = segment.fuzzylabels(times, alts, spds)
    labels = np.array(labels)

    dset = {}
    dset['climb'] = data[labels == 'CL', :]
    dset['cruise'] = data[labels == 'CR', :]
    dset['descend'] = data[labels == 'DE', :]
    dset['ground'] = data[labels == 'GND', :]

    entries = []
    for p, d in dset.iteritems():
        if len(d) < MIN_DATA_SIZA:
            continue

        tdata = np.copy(d)
        tdata[:, 1:] = 0
        cluster.fit(tdata)
        labels = cluster.labels_
        n_clusters = np.unique(labels).size
        # print("n_clusters : %d" % n_clusters)

        for i in range(n_clusters):
            mask = labels == i
            c = d[mask, :]
            if len(c) < MIN_DATA_SIZA:
                continue

            entries.append({
                'icao': icao,
                'phase': p,
                'data': c.tolist()
            })

    if not entries:
        continue

    mcollout.insert(entries)
    print "Progress : %d of %d" % (count, total)
