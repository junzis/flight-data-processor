import argparse
import numpy as np
import pandas as pd
from itertools import cycle
from matplotlib import pyplot as plt
from sklearn import preprocessing
from pymongo import MongoClient
from collections import OrderedDict
from sklearn.cluster import DBSCAN

# Constants
HOST = "localhost"  # MongoDB host
PORT = 27017  # MongoDB port
MIN_DATA_SIZA = 100  # minimal number of data in a flight
CHUNK_SIZE = 50  # number of icaos to be processed in chunks
TEST_FLAG = True  # if this is a test run

# get script arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv", dest="input_csv", required=True, help="decoded adsb csv file"
)
parser.add_argument("--db", dest="db", required=True)
parser.add_argument("--coll", dest="output_coll", help="Flight collection name")
args = parser.parse_args()

mdb = args.db
pos_csv = args.input_csv
flights_coll = args.output_coll

mongo_client = MongoClient(HOST, PORT)

if flights_coll:
    TEST_FLAG = False
    mcollflights = mongo_client[mdb][flights_coll]
    mcollflights.drop()  # clear the segment collection first

print("[1] Querying database.")

df = pd.read_csv(pos_csv)
df.drop_duplicates(subset=["ts"], inplace=True)

# Find all ICAO IDs in the dataset
dfcount = df.groupby("icao").size().reset_index(name="counts")
icaos = dfcount[dfcount.counts > 100].icao.tolist()

print("[2] %d number of valid ICAOs." % len(icaos))

for i in range(0, len(icaos), CHUNK_SIZE):

    print("[3][%d-%d of %d] ICAOs beening processed." % (i, i + CHUNK_SIZE, len(icaos)))

    chunk = icaos[i : i + CHUNK_SIZE]

    print("  [a] fetching records")

    dfchunk = df[df.icao.isin(chunk)]
    ids = dfchunk["icao"].as_matrix()
    lats = dfchunk["lat"].as_matrix()
    lons = dfchunk["lon"].as_matrix()
    alts = dfchunk["alt"].as_matrix()
    spds = dfchunk["spd"].as_matrix()
    hdgs = dfchunk["hdg"].as_matrix()
    rocs = dfchunk["roc"].as_matrix()
    times = dfchunk["ts"].as_matrix()

    #####################################################################
    # Continous fligh path extraction using machine learning algorithms
    #####################################################################

    print("  [b] data scaling")

    # transform the text ids into numbers
    # ------------------------------------
    le = preprocessing.LabelEncoder()
    encoded_ids = le.fit_transform(ids)

    # Apply feature scaling - on altitude, spds, and times
    # ------------------------------------------------------
    mms = preprocessing.MinMaxScaler(feature_range=(0, 1000))
    times_norm = mms.fit_transform(times.reshape((-1, 1)))
    dt = mms.scale_ * 0.5 * 60 * 60  # time interval of 30 mins

    mms = preprocessing.MinMaxScaler(feature_range=(0, 100))
    alts_norm = mms.fit_transform(alts.reshape((-1, 1)))

    print("  [c] creating machine learning dataset.")

    # aggregate data by there ids
    # ----------------------------
    acs = {}
    for i in range(len(ids)):
        if ids[i] not in list(acs.keys()):
            acs[ids[i]] = []

        acs[ids[i]].append(
            [
                times_norm[i],
                alts_norm[i],
                times[i],
                lats[i],
                lons[i],
                int(alts[i]),
                spds[i],
                hdgs[i],
                rocs[i],
            ]
        )

    print("  [d] start clustering, and saving results to DB")

    # Apply clustering method
    # ------------------------
    cluster = DBSCAN(eps=dt, min_samples=MIN_DATA_SIZA)

    acsegs = {}
    total = len(list(acs.keys()))
    count = 0
    print("    * processing ", end=" ")
    for k in list(acs.keys()):
        count += 1
        print(".", end=" ")

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
                c = c[c[:, 0].argsort()]  # sort by ts

                if len(c) > MIN_DATA_SIZA:
                    mcollflights.insert_one(
                        OrderedDict(
                            [
                                ("icao", k),
                                ("ts", c[:, 0].tolist()),
                                ("lat", c[:, 1].tolist()),
                                ("lon", c[:, 2].tolist()),
                                ("alt", c[:, 3].tolist()),
                                ("spd", c[:, 4].tolist()),
                                ("hdg", c[:, 5].tolist()),
                                ("roc", c[:, 6].tolist()),
                            ]
                        )
                    )

        # Plot result
        if TEST_FLAG:
            colorset = cycle(["purple", "green", "red", "blue", "orange"])
            for i, c in zip(list(range(n_clusters)), colorset):
                mask = labels == i
                ts = data[mask, 0].tolist()
                alts = data[mask, 5].tolist()
                if len(alts) > MIN_DATA_SIZA:
                    plt.plot(ts, alts, "w", color=c, marker=".", alpha=1.0)

            plt.xlim([0, 1000])
            plt.draw()
            plt.waitforbuttonpress(-1)
            plt.clf()
    print("")

print()
print("[4] All completed")
