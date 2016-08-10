import sys
from ctypes import *
from scipy.optimize import fsolve, brentq
from scipy.integrate import nquad, quad, romberg, quadrature
from math import radians, degrees
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

TINY = 1.0e-5

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

    rootVal = fsolve(root, 1.0e-3)[0]
    # rootVal = brentq(root, 0.0, 0.6)
    return rootVal

def r0_max_val(r, y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    thP0 = thetaPrime(r, thv, 0.0)
    rExp = -np.power(np.divide(thP0, sig), 2.0*kap)
    lhs = np.divide(y - np.power(y, 5.0 - k), Gk)
    rhs = (np.tan(thv) + r)**2.0*np.exp2(rExp)
    return rhs - lhs

vec_r0Max_val = np.vectorize(r0_max_val)

def fluxG_fullStr(r, y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    thP0 = thetaPrime(r, thv, 0.0)
    exp0 = np.power(np.divide(thP0, sig), 2.0*kap)
    chiVal = np.divide(y - Gk*np.exp2(-exp0)*(np.tan(thv) + r)**2.0, np.power(y, 5.0 - k))
    # if chiVal < 1.0:
        # return 0.0
    # else:
        # return r*intG(y, chiVal)*phiInt(r, kap, thv, sig)
    return r*intG(y, chiVal)*phiInt(r, kap, thv, sig)

vec_fluxG_fullStr = np.vectorize(fluxG_fullStr)

def bounds_yr(kap, sig, thv):
    return [TINY, 1.0]

def bounds_ry(y, kap, sig, thv):
    R0MAX = r0_max(y, kap, sig, thv)
    if R0MAX < 0.0:
        return [0.0, 0.0]
    else:
        return [0.0, R0MAX]

def plot_r0Int(y, kap, sig, thv):
    R0_MAX = r0_max(y, kap, sig, thv)
    if R0_MAX > 0.0:
        # r0s = np.linspace(0.001, R0_MAX, num = 1000)
        r0s = np.logspace(-3, np.log10(R0_MAX), num = 100)
        # vals = np.asarray([fluxG_fullStr(r0, y, kap, sig, thv) for r0 in r0s])
        vals = vec_fluxG_fullStr(r0s, y, kap, sig, thv)
        dat = pd.DataFrame(data = {'r0': r0s, 'int': vals})
        NUM_ROWS = len(dat)
        dat['y'] = np.repeat(y, NUM_ROWS)
        dat['kap'] = np.repeat(kap, NUM_ROWS)
        dat['thv'] = np.repeat(thv, NUM_ROWS)
        # print data.head()
        # plt.plot(r0s, vals, label = str(y))
        # plt.loglog(r0s, vals, label = str(y))
        return(dat)

def plot_r0Max(y, kap, sig, thv):
    r0s = np.linspace(0.0, 1.0, num = 100)
    vals = vec_r0Max_val(r0s, y, kap, sig, radians(thv))
    dat = pd.DataFrame(data = {'r0': r0s, 'int': vals})
    NUM_ROWS = len(dat)
    dat['y'] = np.repeat(y, NUM_ROWS)
    dat['kap'] = np.repeat(kap, NUM_ROWS)
    dat['thv'] = np.repeat(thv, NUM_ROWS)
    # max_val = r0_max(y, kap, sig, thv)
    # dat['r0max'] = np.repeat(max_val, NUM_ROWS)
    # dat['maxval'] = np.repeat(0.0, NUM_ROWS)
    return(dat)
    
def plot_r0Int_grid():
    SIGMA = 2.0
    df_list = []
    max_list = [[[] for x in range(3)] for y in range(3)]
    # i = 0
    for i, KAPPA in enumerate([0.0, 1.0, 10.0]):
        for j, THETA_V in enumerate([0.0, 2.0, 6.0]):
            for y in [TINY, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - TINY]:
            # for y in [TINY, 1.0 - TINY]:
                df = plot_r0Int(y, KAPPA, SIGMA, radians(THETA_V))
                # df = plot_r0Max(y, KAPPA, SIGMA, THETA_V)
                # print df.head()
                df_list.append(df)
                
                # max_val = r0_max(y, KAPPA, SIGMA, radians(THETA_V))
                # max_list[j][i].append(max_val)
                # print KAPPA, THETA_V, y, max_val
            # i += 1
    
    data = pd.concat(df_list)
    # print data.head()
    # plt.figure()
    grid = sns.lmplot(x = 'r0', y = 'int', hue = 'y',
                        col = 'kap', row = 'thv', data = data, markers = '.',
                        palette = 'viridis', fit_reg = False)  # 
    # grid.map(plt.axhline, color = 'red', linestyle = '--')
    # grid.map(plt.scatter, 'r0max', 'maxval')
    grid.set(yscale="log")
    grid.set(xscale="log")
    # grid.set_axis_labels("r0'", "Root Function")
    grid.set_axis_labels("r0'", "r0' Integrand")
    
    # for loc, data in grid.facet_data():
        # # print loc
        # grid.axes[loc[0], loc[1]].scatter(max_list[loc[0]][loc[1]], [0.0 for x in range(7)], marker = 'o')
    axes = grid.axes
    # for i, ax in enumerate(axes.flat):
        # print i, ax.get_xlim()
        # for rm in max_list[i]:
            # ln_ = ax.axvline(x = rm, linestyle = '--', color = 'red')
    
    axes[0, 0].set_ylim(1.0e-9, )
    axes[0, 0].set_xlim(1.0e-3, )
    grid.set_titles('thv = {row_name} | kap = {col_name}')
    plt.show()
    # grid.savefig("r0-Int.png")

def r0_integral():
    SIG = 2.0
    dat_list = []
    for i, KAP in enumerate([0.0, 1.0, 10.0]):
        for j, THETA_V in enumerate([0.0, 2.0, 6.0]):
            THV = radians(THETA_V)
            ys = np.linspace(0.0, 1.0, 100)
            ints = np.zeros(len(ys))
            for index in range(len(ys)):
                YVAL = ys[index]
                R0_MAX = r0_max(YVAL, KAP, SIG, THV)
                if R0_MAX > 0.0:
                    int_val = quad(fluxG_fullStr, 0.0, R0_MAX,
                                    args = (YVAL, KAP, SIG, THV),
                                    epsabs = 1.0e-5)[0]
                    print KAP, THETA_V, YVAL, int_val
                    ints[index] = int_val
            
            loc_df = pd.DataFrame(data = {'y': ys, 'ival': ints})
            N = len(loc_df)
            loc_df['kap'] = np.repeat(KAP, N)
            loc_df['thv'] = np.repeat(THETA_V, N)
            
            dat_list.append(loc_df)
    
    df = pd.concat(dat_list)
    grid = sns.lmplot(x = 'y', y = 'ival', col = 'kap', row = 'thv', data = df,
                        fit_reg = False)
    plt.show()

def main():
    tiny = np.power(10.0, -3.0)
    SIGMA = 2.0
    KAPPA = 1.0
    THETA_V = radians(6.0)
    YVAL = 0.5
    # plot_r0Int_grid()
    r0_integral()

    # # KAPPA = tiny
    # # for kap in range(10):
    # for kap in [0.0, 1.0, 3.0, 10.0]:
        # # KAPPA = np.power(10.0, -float(kap + 1))
        # KAPPA = float(kap)  # + 0.01
        # # print fluxG_fullStr(0.1, 0.5, 1.0, 2.0, radians(6.0))
        # str_int = nquad(fluxG_fullStr, [bounds_ry, bounds_yr], 
                        # args = (KAPPA, SIGMA, THETA_V),
                        # opts = {'epsabs': 1.0e-5})
        # # str_int = quad(fluxG_fullStr, TINY, r0_max(YVAL, KAPPA, SIGMA, THETA_V) - TINY,
                        # # args = (YVAL, KAPPA, SIGMA, THETA_V), epsabs = 1.0e-5)
        # # str_int = romberg(vec_fluxG_fullStr, TINY, r0_max(YVAL, KAPPA, SIGMA, THETA_V) - TINY,
                            # # args = (YVAL, KAPPA, SIGMA, THETA_V), tol = 1.0e-5,
                            # # vec_func = True)
        # # str_int = quadrature(vec_fluxG_fullStr, TINY, r0_max(YVAL, KAPPA, SIGMA, THETA_V) - TINY,
                            # # args = (YVAL, KAPPA, SIGMA, THETA_V), tol = 1.0e-5,
                            # # vec_func = True)
        # print KAPPA, str_int

if __name__ == "__main__":
    sys.exit(int(main() or 0))