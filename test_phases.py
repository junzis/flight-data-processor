import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import flightphase
from extra.filters import Spline

flightphase.plot_logics()

# get a sample data
datadir = os.path.dirname(os.path.realpath(__file__))
dataset = cPickle.load(open(datadir+'/data/test_segment.pkl', 'rb'))

for data in dataset:
    times = np.array(data['ts'])
    times = times - times[0]
    alts = np.array(data['H']) / 0.3048
    vgx = np.array(data['vgx'])
    vgy = np.array(data['vgy'])
    spds = np.sqrt(vgx**2 + vgy**2) / 0.514444
    rocs = np.array(data['vh']) / 0.00508

    labels = flightphase.fuzzylabels(times, alts, spds, rocs)

    colormap = {'GND': 'black', 'CL': 'green', 'CR': 'blue',
                'DE': 'orange', 'LVL': 'purple', 'NA': 'red'}

    colors = [colormap[l] for l in labels]

    fltr = Spline(k=2)
    _, altspl = fltr.filter(times, alts)
    _, spdspl = fltr.filter(times, spds)
    _, rocspl = fltr.filter(times, rocs)

    plt.subplot(311)
    plt.title('press any key to continue to next example...')
    plt.plot(times, altspl, '-', color='k', alpha=0.5)
    plt.scatter(times, alts, marker='.', c=colors, lw=0)
    plt.ylabel('altitude (ft)')

    plt.subplot(312)
    plt.plot(times, spdspl, '-', color='k', alpha=0.5)
    plt.scatter(times, spds, marker='.', c=colors, lw=0)
    plt.ylabel('speed (kt)')

    plt.subplot(313)
    plt.plot(times, rocspl, '-', color='k', alpha=0.5)
    plt.scatter(times, rocs, marker='.', c=colors, lw=0)
    plt.ylabel('roc (fpm)')

    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
