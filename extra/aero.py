"""
Functions for aeronautics in this module
    - physical quantities always in SI units
    - lat,lon,course and heading in degrees

International Standard Atmosphere
    p,rho,T = atmos(h)    # atmos as function of geopotential altitude h [m]
    a = vsound(h)         # speed of sound [m/s] as function of h[m]
    p = pressure(h)       # calls atmos but retruns only pressure [Pa]
    T = temperature(h)    # calculates temperature [K]
    rho = density(h)      # calls atmos but retruns only pressure [Pa]

Speed conversion at altitude h[m] in ISA:
    M   = tas2mach(tas,h)  # true airspeed (tas) to mach number conversion
    tas = mach2tas(M,h)    # true airspeed (tas) to mach number conversion
    tas = eas2tas(eas,h)   # equivalent airspeed to true airspeed, h in [m]
    eas = tas2eas(tas,h)   # true airspeed to equivent airspeed, h in [m]
    tas = cas2tas(cas,h)   # cas  to tas conversion both m/s, h in [m]
    cas = tas2cas(tas,h)   # tas to cas conversion both m/s, h in [m]
    cas = mach2cas(M,h)    # Mach to cas conversion cas in m/s, h in [m]
    M   = cas2mach(cas,h)  # cas to mach copnversion cas in m/s, h in [m]
"""

import numpy as np

kts = 0.514444      # knot -> m/s
ft = 0.3048         # ft -> m
fpm = 0.00508       # ft/min -> m/s
inch = 0.0254       # inch -> m
sqft = 0.09290304   # 1 square foot
nm = 1852.          # nautical mile -> m
lbs = 0.453592      # pound -> kg
g0 = 9.80665        # m/s2, Sea level gravity constant
R = 287.05287       # m2/(s2 x K), gas constant, sea level ISA
p0 = 101325.        # Pa, air pressure, sea level ISA
rho0 = 1.225        # kg/m3, air density, sea level ISA
T0 = 288.15         # K, temperature, sea level ISA
gamma = 1.40        # cp/cv for air
gamma1 = 0.2        # (gamma-1)/2 for air
gamma2 = 3.5        # gamma/(gamma-1) for air
beta = -0.0065      # [K/m] ISA temp gradient below tropopause
r_earth = 6371000.  # m, average earth radius
a0 = 340.293988     # m/s, sea level speed of sound ISA, sqrt(gamma*R*T0)


def atmos(H):
    # H in metres
    T = np.maximum(288.15-0.0065*H, 216.65)
    rhotrop = 1.225*(T/288.15)**4.256848030018761
    dhstrat = np.maximum(0., H-11000.)
    rho = rhotrop * np.exp(-dhstrat / 6341.552161)
    p = rho * R * T
    return p, rho, T


def temperature(H):
    """Temperature at given altitude"""
    p, r, T = atmos(H)
    return T


def pressure(H):
    """Pressure at given altitude"""
    p, r, T = atmos(H)
    return p


def density(H):
    """Air density at given altitude"""
    p, r, T = atmos(H)
    return r


def vsound(H):
    """Speed of sound at given altitude"""
    T = temperature(H)
    a = np.sqrt(gamma*R*T)
    return a


def distance(lat1, lon1, lat2, lon2, H=0):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    d2r = np.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * d2r
    phi2 = (90.0 - lat2) * d2r

    # theta = longitude
    theta1 = lon1 * d2r
    theta2 = lon2 * d2r

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
           np.cos(phi1)*np.cos(phi2))
    arc = np.arccos(cos)

    dist = arc * (r_earth + H)   # meters, radius of earth

    return dist


def bearing(lat1, lon1, lat2, lon2):
    x = np.sin(lon2-lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) \
        - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    return bearing


###########################################################
# Speed conversions, (h in meters)
###########################################################
def tas2mach(tas, h):
    """True Airspeed to Mach number"""
    a = vsound(h)
    M = tas/a
    return M


def mach2tas(M, h):
    """Mach number to True Airspeed"""
    a = vsound(h)
    tas = M*a
    return tas


def eas2tas(eas, h):
    """Equivalent Airspeed to True Airspeed"""
    rho = density(h)
    tas = eas * np.sqrt(rho0/rho)
    return tas


def tas2eas(tas, h):
    """True Airspeed to Equivalent Airspeed"""
    rho = density(h)
    eas = tas * np.sqrt(rho/rho0)
    return eas


def cas2tas(cas, h):
    """Calibrated Airspeed to True Airspeed"""
    p, rho, T = atmos(h)
    qdyn = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas = np.sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    return tas


def tas2cas(tas, h):
    """True Airspeed to Calibrated Airspeed"""
    p, rho, T = atmos(h)
    qdyn = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas = np.sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
    return cas


def mach2cas(M, h):
    """Mach number to Calibrated Airspeed"""
    tas = mach2tas(M, h)
    cas = tas2cas(tas, h)
    return cas


def cas2mach(cas, h):
    """Calibrated Airspeed to Mach number"""
    tas = cas2tas(cas, h)
    M = tas2mach(tas, h)
    return M
