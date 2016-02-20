import pickle
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from scipy import stats
from filters import SavitzkyGolay

# margin of outputs rules determin if a state is a valid state
STATE_DIFF_MARGIN = 0.2

# logic states
alt_range = np.arange(0, 40000, 1)
dh_range = np.arange(-20, 20, 0.1)
spd_range = np.arange(0, 600, 1)
states = np.arange(0, 5, 0.01)

alt_gnd = fuzz.zmf(alt_range, 0, 200)
alt_lo = fuzz.gaussmf(alt_range, 10000, 10000)
alt_hi = fuzz.gaussmf(alt_range, 30000, 10000)

dh_zero = fuzz.gaussmf(dh_range, 0, 5)
dh_plus = fuzz.sigmf(dh_range, 6, 1)
dh_minus = fuzz.sigmf(dh_range, -6, -1)

spd_hi = fuzz.gaussmf(spd_range, 600, 200)
spd_md = fuzz.gaussmf(spd_range, 100, 100)
spd_lo = fuzz.gaussmf(spd_range, 0, 50)

state_ground = fuzz.gaussmf(states, 1, 0.1)
state_climb = fuzz.gaussmf(states, 2, 0.1)
state_descend = fuzz.gaussmf(states, 3, 0.1)
state_cruise = fuzz.gaussmf(states, 4, 0.1)


# Visualize these universes and membership functions
def plot_logics():
    plt.figure(figsize=(10, 8))

    plt.subplot(411)
    plt.plot(alt_range, alt_gnd, color='black', lw=2, label='Ground')
    plt.plot(alt_range, alt_lo, color='green', lw=2, label='Low')
    plt.plot(alt_range, alt_hi, color='blue', lw=2, label='High')
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Altitude (ft)')
    plt.yticks([0, 1])
    plt.legend(prop={'size': 11})

    plt.subplot(412)
    plt.plot(dh_range, dh_zero, color='steelblue', lw=2, label='Zero')
    plt.plot(dh_range, dh_plus, color='green', lw=2, label='Positive')
    plt.plot(dh_range, dh_minus, color='orange', lw=2, label='Negative')
    plt.ylim([-0.05, 1.05])
    plt.ylabel('RoC (ft/s)')
    plt.yticks([0, 1])
    plt.legend(prop={'size': 11})

    plt.subplot(413)
    plt.plot(spd_range, spd_hi, color='blue', lw=2, label='High')
    plt.plot(spd_range, spd_md, color='maroon', lw=2, label='Midium')
    plt.plot(spd_range, spd_lo, color='black', lw=2, label='Low')
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Speed (kn)')
    plt.yticks([0, 1])
    plt.legend(prop={'size': 11})

    plt.subplot(414)
    plt.plot(states, state_ground, color='black', lw=2, label='Ground')
    plt.plot(states, state_climb, color='green', lw=2, label='Climb')
    plt.plot(states, state_descend, color='orange', lw=2, label='Descend')
    plt.plot(states, state_cruise, color='blue', lw=2, label='Cruise')
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Flight Phases')
    plt.yticks([0, 1])
    plt.legend(prop={'size': 11})
    plt.show()


def fuzzylabels(ts, alts, spds, twindow=60):
    '''
    Fuzzy logic to determine the segments of the flight data
    segments are: ground [GND], climb [CL], descend [DE], cruise [CR].

    Default time window is 60 second.
    '''

    if len(ts) != len(alts):
        raise RuntimeError('input ts and alts must have same length')

    fltr = SavitzkyGolay(order=1, window_size=11)
    _, alts = fltr.filter(ts, alts)
    _, spds = fltr.filter(ts, spds)

    labels = ['NA'] * len(ts)

    # seprate data into chunks by window size
    idxchunks = []
    tchunks = []
    altchunks = []
    spdchunks = []
    idxchk = []
    tchk = []
    altchk = []
    spdchk = []
    t0 = ts[0]
    c = 1   # chunk number
    idx = 0
    for t, alt, spd in zip(ts, alts, spds):
        if t > t0 + c * twindow:
            if len(idxchk) > 2:
                idxchunks.append(idxchk)
                tchunks.append(tchk)
                altchunks.append(altchk)
                spdchunks.append(spdchk)

            idxchk = []
            tchk = []
            altchk = []
            spdchk = []

            idxchk.append(idx)
            tchk.append(t)
            altchk.append(alt)
            spdchk.append(spd)

            c += 1
        else:
            idxchk.append(idx)
            tchk.append(t)
            altchk.append(alt)
            spdchk.append(spd)
        idx += 1
    # add remaining parts from last iteration
    idxchunks.append(idxchk)
    tchunks.append(tchk)
    altchunks.append(altchk)

    for idxchk, tchk, altchk, spdchk \
            in zip(idxchunks, tchunks, altchunks, spdchunks):

        alt = np.mean(altchk)
        spd = np.mean(spdchk)

        if alt > alt_range[-1]:
            alt = alt_range[-1]
        if alt < alt_range[0]:
            alt = alt_range[0]

        if spd > spd_range[-1]:
            spd = spd_range[-1]
        if spd < spd_range[0]:
            spd = spd_range[0]

        lr = stats.linregress(xrange(len(altchk)), altchk)
        dh = lr[0]

        if dh > dh_range[-1]:
            dh = dh_range[-1]
        if dh < dh_range[0]:
            dh = dh_range[0]

        alt_level_gnd = fuzz.interp_membership(alt_range, alt_gnd, alt)
        alt_level_lo = fuzz.interp_membership(alt_range, alt_lo, alt)
        alt_level_hi = fuzz.interp_membership(alt_range, alt_hi, alt)

        spd_level_hi = fuzz.interp_membership(spd_range, spd_hi, spd)
        spd_level_md = fuzz.interp_membership(spd_range, spd_md, spd)
        spd_level_lo = fuzz.interp_membership(spd_range, spd_lo, spd)

        dh_level_zero = fuzz.interp_membership(dh_range, dh_zero, dh)
        dh_level_plus = fuzz.interp_membership(dh_range, dh_plus, dh)
        dh_level_minus = fuzz.interp_membership(dh_range, dh_minus, dh)

        # print alt_level_gnd, alt_level_lo, alt_level_hi
        # print dh_level_zero, dh_level_plus, dh_level_minus
        # print spd_level_hi, spd_level_md, spd_level_lo

        rule_ground = min(alt_level_gnd, spd_level_lo)
        state_activate_ground = np.fmin(rule_ground, state_ground)

        rule_climb = min(alt_level_lo, dh_level_plus, spd_level_md)
        state_activate_climb = np.fmin(rule_climb, state_climb)

        rule_descend = min(alt_level_lo, dh_level_minus, spd_level_md)
        state_activate_descend = np.fmin(rule_descend, state_descend)

        rule_cruise = min(alt_level_hi, dh_level_zero, spd_level_hi)
        state_activate_cruise = np.fmin(rule_cruise, state_cruise)

        aggregated = np.fmax(state_activate_ground,
                             np.fmax(state_activate_climb,
                                     np.fmax(state_activate_descend,
                                             state_activate_cruise)))

        # print aggregated

        state_raw = fuzz.defuzz(states, aggregated, 'lom')
        state = int(round(state_raw))

        state_lable_map = {1: 'GND', 2: 'CL', 3: 'DE', 4: 'CR'}

        for i in idxchk:
            labels[i] = state_lable_map[state]

    return labels


if __name__ == '__main__':
    # plot_logics()

    print "Runing Fuzzy Segmenting rocess on test data."

    # get a sample data
    data = pickle.load(open('../data/full-flight-1.pkl', 'rb'))
    data = np.array(data)

    times = data[:, 0].astype(int)
    times = times - times[0]
    lats = data[:, 1]
    lons = data[:, 2]
    alts = data[:, 3].astype(int)
    spds = data[:, 4].astype(int)

    labels = fuzzylabels(times, alts, spds, twindow=60)

    labels = fuzzylabels(times, alts, spds)

    colormap = {'GND': 'black', 'CL': 'green', 'CR': 'blue', 'DE': 'orange',
                'NA': 'red'}

    colors = [colormap[l] for l in labels]

    plt.subplot(121)
    plt.scatter(times, alts, marker='.', c=colors, lw=0)

    plt.subplot(122)
    plt.scatter(times, spds, marker='.', c=colors, lw=0)

    plt.show()
