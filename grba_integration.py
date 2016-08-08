import sys
from ctypes import *
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import nquad
from math import radians, degrees

grbaint = cdll.LoadLibrary("Release/grba_integration.dll")
phiInt = grbaint.phiInt
phiInt.restype = c_double
phiInt.argtypes = [c_double, c_double, c_double, c_double]
thp = grbaint.thetaPrime
thp.restype = c_double
thp.argtypes = [c_double, c_double, c_double]

def intG(y, chi, k = 0.0, p = 2.2):
    bG = (1.0 - p)/2.0
    ys = np.power(y, 0.5*(bG*(4.0 - k) + 4.0 - 3.0*k))
    chis = np.power(chi, np.divide(7.0*k - 23.0 + bG*(13.0 + k), 6.0*(4.0 - k)))
    factor = np.power((7.0 - 2.0*k)*chi*np.power(y, 4.0 - k) + 1.0, bG - 2.0)
    return ys*chis*factor

def fluxG(y, chi, k = 0.0, p = 2.2):
    Ck = (4.0 - k)*np.power(5.0 - k, np.divide(k - 5.0, 4.0 - k))
    cov = np.divide(np.power(y, 5.0 - k), 2.0*Ck)
    return 2.0*np.pi*cov*intG(y, chi, k, p)

def thetaPrime(r, thv, phi):
    # top = r*(np.cos(thv)**2.0 - 0.25*np.sin(2.0*thv)**2.0*np.cos(phi)**2.0)**2.0
    # bot = 1.0 + 0.5*r*np.sin(2.0*thv)*np.cos(phi)
    # return np.divide(top, bot)
    return thp(r, thv, phi)

def r0_max(y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    def root(rm):
        thP0 = thetaPrime(rm, thv, 0.0)
        rExp = -np.power(np.divide(thP0, sig), 2.0*kap)
        lhs = np.divide(y - np.power(y, 5.0 - k), Gk)
        rhs = (np.tan(thv) + rm)**2.0*np.exp2(rExp)
        return rhs - lhs

    rootVal = fsolve(root, 0.4)[0]
    return rootVal

def fluxG_fullStr(r, y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    thP0 = thetaPrime(r, thv, 0.0)
    exp0 = np.power(np.divide(thP0, sig), 2.0*kap)
    chiVal = np.divide(y - Gk*np.exp2(-exp0)*(np.tan(thv) + r)**2.0, np.power(y, 5.0 - k))
    if chiVal < 1.0:
        return 0.0
    else:
        return r*intG(y, chiVal)*phiInt(r, kap, degrees(thv), sig)

def bounds_yr(kap, sig, thv):
    return [0.1, 1.0]

def bounds_ry(y, kap, sig, thv):
    return [0.1, r0_max(y, kap, sig, thv)]

def main():
    tiny = np.power(10.0, -3.0)
    SIGMA = 2.0
    #KAPPA = tiny
    for kap in range(10):
        # KAPPA = np.power(10.0, -float(kap + 1))
        KAPPA = float(kap) + 0.01
        # print fluxG_fullStr(0.1, 0.5, 1.0, 2.0, radians(6.0))
        str_int = nquad(fluxG_fullStr, [bounds_ry, bounds_yr], 
                        args = (KAPPA, SIGMA, radians(6.0)))
        print str_int

if __name__ == "__main__":
    sys.exit(int(main() or 0))