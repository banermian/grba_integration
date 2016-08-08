import sys
from ctypes import *
from scipy.optimize import fsolve
from scipy.integrate import nquad
from math import radians, degrees
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

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
        return r*intG(y, chiVal)*phiInt(r, kap, thv, sig)

vec_fluxG_fullStr = np.vectorize(fluxG_fullStr)

def bounds_yr(kap, sig, thv):
    return [0.1, 1.0]

def bounds_ry(y, kap, sig, thv):
    return [0.1, r0_max(y, kap, sig, thv)]

def plot_r0Int(y, kap, sig, thv):
    R0_MAX = r0_max(y, kap, sig, thv)
    # r0s = np.linspace(0.001, R0_MAX, num = 1000)
    r0s = np.logspace(-3, np.log10(R0_MAX), num = 100)
    # vals = np.asarray([fluxG_fullStr(r0, y, kap, sig, thv) for r0 in r0s])
    vals = vec_fluxG_fullStr(r0s, y, kap, sig, radians(thv))
    dat = pd.DataFrame(data = {'r0': r0s, 'int': vals})
    NUM_ROWS = len(dat)
    dat['y'] = np.repeat(y, NUM_ROWS)
    dat['kap'] = np.repeat(kap, NUM_ROWS)
    dat['thv'] = np.repeat(thv, NUM_ROWS)
    # print data.head()
    # plt.plot(r0s, vals, label = str(y))
    # plt.loglog(r0s, vals, label = str(y))
    return(dat)
    
def plot_r0Int_grid():
    SIGMA = 2.0
    df_list = []
    for KAPPA in [0.0, 1.0, 10.0]:
        for THETA_V in [0.0, 2.0, 6.0]:
            for y in [0.1, 0.25, 0.5, 0.75, 0.9]:
                df = plot_r0Int(y, KAPPA, SIGMA, THETA_V)
                # print df.head()
                df_list.append(df)
    
    data = pd.concat(df_list)
    # plt.figure()
    grid = sns.lmplot(x = 'r0', y = 'int', hue = 'y',
                        col = 'kap', row = 'thv', data = data, fit_reg = False,
                        palette = 'viridis')  # 
    grid.set(yscale="log")
    grid.set(xscale="log")
    axes = grid.axes
    axes[0, 0].set_ylim(1.0e-9, )
    axes[0, 0].set_xlim(1.0e-3, )
    grid.set_titles('thv = {row_name} | kap = {col_name}')
    # plt.show()
    grid.savefig("r0-integrand.png")

def main():
    tiny = np.power(10.0, -3.0)
    SIGMA = 2.0
    KAPPA = 1.0
    THETA_V = radians(6.0)
    plot_r0Int_grid()

    # #KAPPA = tiny
    # for kap in range(10):
        # # KAPPA = np.power(10.0, -float(kap + 1))
        # KAPPA = float(kap) + 0.01
        # # print fluxG_fullStr(0.1, 0.5, 1.0, 2.0, radians(6.0))
        # str_int = nquad(fluxG_fullStr, [bounds_ry, bounds_yr], 
                        # args = (KAPPA, SIGMA, radians(6.0)))
        # print str_int

if __name__ == "__main__":
    sys.exit(int(main() or 0))