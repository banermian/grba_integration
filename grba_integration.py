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
engProf = grbaint.energyProfile
engProf.restype = c_double
engProf.argtypes = [c_double, c_double, c_double]
fluxG_cFunc = grbaint.fluxWrap
fluxG_cFunc.restype = c_double
fluxG_cFunc.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double]

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

def r_max(phi, r0, kap, sig, thv):
    def rootR(r):
        thp = thetaPrime(r, thv, phi)
        eng = engProf(thp, sig, kap)
        lhs = eng*(np.power(r, 2) + 2.0*r*np.tan(thv)*np.cos(phi) + np.power(np.tan(thv), 2))
        thp0 = thetaPrime(r, thv, 0.0)
        eng0 = engProf(thp0, sig, kap)
        rhs = np.power(r0 + np.tan(thv), 2)*eng0
        return lhs - rhs
    
    rootValR = fsolve(rootR, r0)[0]
    return rootValR

def r0_max(y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    def rootR0(rm):
        thP0 = thetaPrime(rm, thv, 0.0)
        rExp = -np.power(np.divide(thP0, sig), 2.0*kap)
        lhs = np.divide(y - np.power(y, 5.0 - k), Gk)
        rhs = (np.tan(thv) + rm)**2.0*np.exp2(rExp)
        return rhs - lhs

    rootValR0 = fsolve(rootR0, 1.0e-3)[0]
    # rootValR0 = brentq(root, 0.0, 0.6)
    return rootValR0

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
    try:
        return r*intG(y, chiVal)*phiInt(r, kap, thv, sig)
    except WindowsError as we:
        print kap, thv, y, r, intG(y, chiVal), we.args[0]
    except:
        print "Unhandled Exception"

vec_fluxG_fullStr = np.vectorize(fluxG_fullStr)

def fluxG_fullStr_cFunc(r, y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    try:
        return fluxG_cFunc(y, r, kap, sig, thv, gA, k, p)
    except WindowsError as we:
        print we.args[0]
    except:
        print "Unhandled Exception"

vec_fluxG_fullStr_cFunc = np.vectorize(fluxG_fullStr_cFunc)

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
        r0s = np.linspace(0.0, R0_MAX, num = 100)
        # r0s = np.logspace(-3, np.log10(R0_MAX), num = 100)
        vals = vec_fluxG_fullStr(r0s, y, kap, sig, thv)
        dat = pd.DataFrame(data = {'r0': r0s, 'int': vals})
        NUM_ROWS = len(dat)
        dat['y'] = np.repeat(y, NUM_ROWS)
        dat['kap'] = np.repeat(kap, NUM_ROWS)
        dat['thv'] = np.repeat(thv, NUM_ROWS)
        # print data.head()
        return(dat)

def plot_r0Int_cTest(y, kap, sig, thv):
    R0_MAX = r0_max(y, kap, sig, thv)
    if R0_MAX > 0.0:
        r0s = np.logspace(-3, np.log10(R0_MAX), num = 100)
        vals = vec_fluxG_fullStr(r0s, y, kap, sig, thv)
        cVals = vec_fluxG_fullStr_cFunc(r0s, y, kap, sig, thv)
        lab = np.repeat("Python", len(vals))
        clab = np.repeat("C++", len(cVals))
        dat = pd.DataFrame(data = {'r0': r0s, 'int': vals, 'lab': lab})
        cdat = pd.DataFrame(data = {'r0': r0s, 'int': cVals, 'lab': clab})
        full_dat = pd.concat([dat, cdat])
        NUM_ROWS = len(full_dat)
        full_dat['kap'] = np.repeat(kap, NUM_ROWS)
        full_dat['thv'] = np.repeat(degrees(thv), NUM_ROWS)
        
        return(full_dat)

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

def plot_r0Int_grid_cTest(y):
    SIGMA = 2.0
    # YVAL = TINY
    df_list = []
    for i, KAPPA in enumerate([0.0, 1.0, 10.0]):
        for j, THETA_V in enumerate([0.0, 2.0, 6.0]):
            print KAPPA, THETA_V
            df = plot_r0Int_cTest(y, KAPPA, SIGMA, radians(THETA_V))
            df_list.append(df)
    
    data = pd.concat(df_list)
    print data
    grid = sns.lmplot(x = 'r0', y = 'int', hue = 'lab',
                        col = 'kap', row = 'thv', data = data, markers = 'o',
                        palette = 'viridis', fit_reg = False)
    grid.set(yscale="log")
    grid.set(xscale="log")
    axes = grid.axes
    # axes[0, 0].set_ylim(1.0e-9, )
    axes[0, 0].set_xlim(1.0e-3, )
    plt.show()
    
def plot_r0Int_grid():
    SIGMA = 2.0
    df_list = []
    # max_list = [[[] for x in range(3)] for y in range(3)]
    # i = 0
    for i, KAPPA in enumerate([0.0, 1.0, 10.0]):
        for j, THETA_V in enumerate([0.0, 2.0, 6.0]):
            for y in [TINY, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - TINY]:
            # for y in [TINY, 1.0 - TINY]:
                # print KAPPA, THETA_V, y
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
    # grid.set(xscale="log")
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
    axes[0, 0].set_xlim(0.0, )
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

def plot_rMaxPhi_grid(y, kap, sig, thv):
    R0_MAX = r0_max(y, kap, sig, thv)
    r0s = np.linspace(0.0, R0_MAX, num = 100)
    phis = np.linspace(0.0, 2.0*np.pi, num = 100)
    vec_r_max = np.vectorize(r_max)
    rs = vec_r_max(phis, r0s, kap, sig, thv)
    R, P = np.meshgrid(r0s, phis)
    RM = vec_r_max(P, R, kap, sig, thv)
    RNORM = np.divide(RM, r0s)
    # df = pd.DataFrame(data = {'r0': r0s, 'phi': phis, 'r': rs})
    # df_piv = df.pivot(index = 'phi', columns = 'r0', values = 'r')
    # print df_piv.head()
    df = pd.DataFrame(data = RNORM, index = np.round(np.divide(phis, np.pi), decimals = 1), columns = np.round(r0s, decimals = 3))
    ax = sns.heatmap(df, xticklabels = 10, yticklabels = 25)
    plt.xticks(rotation = 90)
    plt.show()
    
    # plt.figure()
    # plt.pcolormesh(R, P, RM)
    # plt.show()

def main():
    tiny = np.power(10.0, -3.0)
    SIGMA = 2.0
    KAPPA = 0.0
    THETA_V = radians(2.0)
    YVAL = 0.9
    
    for KAP in [0.0, 1.0, 10.0]:
        for THV in [0.0, 2.0, 6.0]:
            # R0MAX = r0_max(YVAL, KAP, SIGMA, radians(THV))
            # step = np.linspace(0.0, R0MAX, num = 100, retstep = True)[1]
            # print KAP, THV, R0MAX, step
    
            plot_rMaxPhi_grid(YVAL, KAP, SIGMA, radians(THV))
    
    # plot_r0Int_grid()
    # plot_r0Int_grid_cTest(0.9)
    # r0_integral()

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