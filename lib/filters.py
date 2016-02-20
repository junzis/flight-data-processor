import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.interpolate import UnivariateSpline


class BaseFilter(object):
    def __init__(self, i=False):
        self.interpolate = i

    def sortxy(self, X, Y):
        XY = zip(X, Y)
        XY.sort(key=lambda t: t[0])
        X1, Y1 = zip(*XY)
        return np.array(X1), np.array(Y1)

    def simplefill(self, X, Y):
        ''' Fill the missing data with closest previous data each second '''

        X, Y = self.sortxy(X, Y)

        X = np.array(X)
        Y = np.array(Y)

        Xfull = range(int(X[0]), int(X[-1]+1))
        Yfull = []

        y = 0
        for x in Xfull:
            try:
                i = np.where(X == x)[0][0]
                y = Y[i]
            except:
                pass

            Yfull.append(y)

        return np.array(Xfull), np.array(Yfull)


class SavitzkyGolay(BaseFilter):
    """
    SavitzkyGolay Filter

    Parameters
    ----------
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0
        means only smoothing)

    """
    def __init__(self, window_size=11, order=2, deriv=0, i=False):
        super(SavitzkyGolay, self).__init__(i=i)

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("win size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("win size is too small for the polynomials order")

        self.window_size = window_size
        self.order = order
        self.deriv = deriv

    def filter(self, X, Y):
        if self.interpolate:
            X, Y = self.simplefill(X, Y)
        else:
            X, Y = self.sortxy(X, Y)

        order_range = range(self.order+1)
        half_window = (self.window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range]
                    for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[self.deriv]
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = Y[0] - np.abs(Y[1:half_window+1][::-1] - Y[0])
        lastvals = Y[-1] + np.abs(Y[-half_window-1:-1][::-1] - Y[-1])
        Y1 = np.concatenate((firstvals, Y, lastvals))
        Y2 = np.convolve(m, Y1, mode='valid')

        return X, Y2


class Kalman(BaseFilter):
    """
    Kalman Filter

    Parameters
    ----------
    Q : float (default: 1e-7)
        process variance
    R : float (default: 0.1**3)
        estimate of measurement variance
    """
    def __init__(self, R=0.00001, Q=1e-7, i=False):
        super(Kalman, self).__init__(i=i)
        self.R = R
        self.Q = Q

    def filter(self, X, Y):
        if self.interpolate:
            X, Y = self.simplefill(X, Y)
        else:
            X, Y = self.sortxy(X, Y)

        # intial parameters
        n_iter = len(Y)
        z = Y

        # allocate space for arrays
        Y1 = np.zeros((n_iter, ))           # a posteri estimate of Y
        P = np.zeros((n_iter, ))            # a posteri error estimate
        Y1minus = np.zeros((n_iter, ))      # a priori estimate of Y
        Pminus = np.zeros((n_iter, ))       # a priori error estimate
        K = np.zeros((n_iter, ))            # gain or blending factor

        # intial guesses
        Y1[0] = 0.0
        P[0] = 1.0

        for k in range(1, n_iter):
            # time update
            Y1minus[k] = Y1[k-1]
            Pminus[k] = P[k-1] + self.Q

            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + self.R)
            Y1[k] = Y1minus[k] + K[k] * (z[k] - Y1minus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return X, Y1


class Spline(BaseFilter):
    """
    Spline smoothing

    """
    def __init__(self, i=False):
        super(Spline, self).__init__(i=i)

    def kernel(self, series, sigma=3):
        '''Apply a gaussian kernel to our data, and get the variances'''
        # http://www.nehalemlabs.net/prototype/blog/2014/04/12/
        #    how-to-fix-scipys-interpolating-spline-default-behavior/
        series = np.asarray(series)
        b = gaussian(25, sigma)
        averages = filters.convolve1d(series, b/b.sum())
        variances = filters.convolve1d(np.power(series-averages, 2), b/b.sum())
        variances[variances == 0] = 1
        return averages, variances

    def filter(self, X, Y):
        X, Y = self.sortxy(X, Y)

        # using gaussian kernel to get a better variances
        averages, variances = self.kernel(Y)
        spl = UnivariateSpline(X, Y, k=1, w=1/np.sqrt(variances))

        if self.interpolate:
            xmax = X[-1]
            Xfull = np.arange(xmax)
            Yfull = spl(Xfull)
            return Xfull, Yfull
        else:
            Y1 = spl(X)
            return X, Y1


class TWF(BaseFilter):
    """
    Time-based weighted filter
    input X is the time stamps of Y
    """
    def __init__(self, window_size=10):
        super(TWF, self).__init__()
        self.window_size = window_size

    def filter(self, X, Y):
        X, Y = self.sortxy(X, Y)

        YF = np.zeros(Y.shape)
        YF[0] = Y[0]
        YF[1] = Y[1]

        for i in range(2, len(X)):
            if i < self.window_size:
                y = (np.average(YF[:i-1]) + Y[i]) / 2.0
            else:
                Xwin = X[i-self.window_size: i-1][::-1]
                Ywin = YF[i-self.window_size: i-1][::-1]
                dXwin = Xwin[1:] - Xwin[:-1]
                yw = (Ywin[0] + np.sum(1./dXwin * Ywin[1:])) / \
                     (1 + np.sum(1.0/dXwin))
                y = (yw + Y[i]) / 2.0
            YF[i] = y
        return X, np.array(YF)



# following are test functions for previous method
def filterplot(ts, alts, spds, fltr, fltrname):
    ts_f, alts_f = fltr.filter(ts, alts)
    ts_f, spds_f = fltr.filter(ts, spds)

    plt.suptitle(fltrname)
    plt.subplot(2, 1, 1)
    plt.plot(ts, alts, '.-', color='steelblue', alpha=0.3, lw=2)
    plt.plot(ts_f, alts_f, '-', color='red')
    plt.ylim([0, 40000])
    plt.ylabel('Altitude (ft)')
    plt.xlabel('time (s)')

    plt.subplot(2, 1, 2)
    plt.plot(ts, spds, '.-', color='sienna', alpha=0.3, lw=2)
    plt.plot(ts_f, spds_f, '-', color='red')
    plt.ylim([0, 700])
    plt.ylabel('Speed (kt)')
    plt.xlabel('time (s)')


if __name__ == '__main__':
    dataset = pickle.load(open('data/dataset_test.pkl', 'rb'))

    sg = SavitzkyGolay(window_size=31, order=1, i=False)
    km = Kalman()
    spl = Spline()

    for data in [dataset[11], dataset[18]]:
        # filterplot(data['ts'], data['alts'], data['spds'],
        #            sg, 'SavitzkyGolay')
        # filterplot(data['ts'], data['alts'], data['spds'], km, 'Kalman')
        filterplot(data['ts'], data['alts'], data['spds'], spl, 'Spline')

        plt.draw()
        plt.waitforbuttonpress(-1)
        plt.clf()
