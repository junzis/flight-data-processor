import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
from pymongo import MongoClient
from mpl_toolkits.basemap import Basemap
import flightphase


def fill_nan(A):
    # interpolate to fill nan values
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], fill_value="extrapolate")
    B = np.where(np.isfinite(A), A, f(inds))
    return B


# get script arguments
parser = argparse.ArgumentParser()

parser.add_argument("--db", dest="db", required=True)
parser.add_argument(
    "--coll", dest="coll", required=True, help="Flights data collection name"
)
args = parser.parse_args()

DB = args.db
COLL = args.coll

# Configuration for the database
mongo_client = MongoClient("localhost", 27017)
mcoll = mongo_client[DB][COLL]

plt.figure(figsize=(10, 4))

res = mcoll.find()
for r in res:
    icao = r["icao"]
    print(r["_id"], icao)

    df = pd.DataFrame(r)
    df.drop_duplicates(subset=["ts"], inplace=True)

    times = np.array(df["ts"])
    times = times - times[0]
    lats = np.array(df["lat"])
    lons = np.array(df["lon"])

    if "alt" in r:
        alts = np.array(df["alt"])
        spds = np.array(df["spd"])
        rocs = np.array(df["roc"])
    elif "H" in r:
        Hs = np.array(df["H"])
        vgxs = np.array(df["vgx"])
        vgys = np.array(df["vgy"])
        vhs = np.array(df["vh"])
        alts = Hs / 0.3048
        spds = np.sqrt(vgxs ** 2 + vgys ** 2) / 0.5144
        rocs = vhs / 0.00508

    try:
        labels = flightphase.fuzzylabels(times, alts, spds, rocs)
    except:
        continue

    colormap = {
        "GND": "black",
        "CL": "green",
        "CR": "blue",
        "DE": "orange",
        "LVL": "purple",
        "NA": "red",
    }

    colors = [colormap[l] for l in labels]

    # setup mercator map projection.
    plt.suptitle("press any key to continue to next example...")

    plt.subplot(121)
    m = Basemap(
        llcrnrlon=min(lons) - 2,
        llcrnrlat=min(lats) - 2,
        urcrnrlon=max(lons) + 2,
        urcrnrlat=max(lats) + 2,
        resolution="l",
        projection="merc",
    )
    m.fillcontinents()
    # plot SIL as a fix point
    latSIL = 51.989884
    lonSIL = 4.375374
    m.plot(lonSIL, latSIL, latlon=True, marker="o", c="red", zorder=9)
    m.scatter(lons, lats, latlon=True, marker=".", c=colors, lw=0, zorder=10)

    plt.subplot(122)
    plt.scatter(times, alts, marker=".", c=colors, lw=0)
    plt.ylabel("altitude (ft)")

    # plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
