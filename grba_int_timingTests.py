import sys
import timeit
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ctypes import *
from scipy.optimize import root, fsolve, brentq
from scipy.integrate import quad, romberg, quadrature, simps
from math import radians, degrees
from grba_int import *

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica Neue UltraLight'
mpl.rcParams['font.variant'] = 'small-caps'
mpl.rcParams['font.size'] = 21
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 17
mpl.rcParams['legend.numpoints'] = 1

# plt.style.use('seaborn-whitegrid')
sns.set_style('ticks', {'legend.frameon': True})

grbaint = cdll.LoadLibrary("Release/grba_integration.dll")
thp = grbaint.thetaPrime
thp.restype = c_double
thp.argtypes = [c_double, c_double, c_double]
engProf = grbaint.energyProfile
engProf.restype = c_double
engProf.argtypes = [c_double, c_double, c_double]

def thetaPrime(r, thv, phi):
    # top = r*(np.cos(thv)**2.0 - 0.25*np.sin(2.0*thv)**2.0*np.cos(phi)**2.0)**2.0
    # bot = 1.0 + 0.5*r*np.sin(2.0*thv)*np.cos(phi)
    # return np.divide(top, bot)
    return thp(r, thv, phi)

def r0_max(y, kap, sig, thv, gA = 1.0, k = 0.0, p = 2.2):
    Gk = (4.0 - k)*gA**2.0
    def rootR0(rm):
        thP0 = thetaPrime(rm, thv, 0.0)
        rExp = -np.power(np.divide(thP0, sig), 2.0*kap)
        lhs = np.divide(y - np.power(y, 5.0 - k), Gk)
        rhs = (np.tan(thv) + rm)**2.0*np.exp2(rExp)
        return rhs - lhs

    rootValR0 = fsolve(rootR0, 1.0e-5)[0]
    # rootValR0 = brentq(root, 0.0, 0.6)
    return rootValR0

def root_fun(r, r0, phi, kap, sig, thv):
    thp = thetaPrime(r, thv, phi)
    eng = engProf(thp, sig, kap)
    lhs = eng*(np.power(r, 2.0) + 2.0*r*np.tan(thv)*np.cos(phi) + np.power(np.tan(thv), 2.0))
    thp0 = thetaPrime(r, thv, 0.0)
    eng0 = engProf(thp0, sig, kap)
    rhs = np.power(r0 + np.tan(thv), 2.0)*eng0
    return lhs - rhs

def root_jac(r, r0, phi, kap, sig, thv):
    thp = thetaPrime(r, thv, phi)
    first = r + np.tan(thv)*np.cos(phi)
    second = np.power(r, 2.0) + 2.0*r*np.tan(thv)*np.cos(phi) + np.power(np.tan(thv), 2.0)
    frac = (kap*np.log(2.0)*np.power(thp / sig, 2.0*kap)) / (r*(1.0 + 0.5*r*np.sin(2.0*thv)*np.cos(phi)))
    exponent = 2.0*engProf(thp, sig, kap)
    return (first - second*frac)*exponent

def intG(y, chi, k = 0.0, p = 2.2):
    bG = (1.0 - p)/2.0
    ys = np.power(y, 0.5*(bG*(4.0 - k) + 4.0 - 3.0*k))
    chis = np.power(chi, np.divide(7.0*k - 23.0 + bG*(13.0 + k), 6.0*(4.0 - k)))
    factor = np.power((7.0 - 2.0*k)*chi*np.power(y, 4.0 - k) + 1.0, bG - 2.0)
    return ys*chis*factor

def root_test():
    TINY = np.power(10.0, -9.0)
    SIGMA = 2.0
    for YVAL in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
    # for YVAL in [0.5]:
        ms = []
        ts = []
        rs = []
        ks = []
        ths = []
        cs = []
        for KAP in [0.0, 1.0, 10.0]:
            for THV in [1.0, 2.0, 6.0]:
                THETA_V = radians(THV)
                R0MAX = r0_max(YVAL, KAP, SIGMA,THETA_V)
                if R0MAX > 0.0:
                    r0s = np.linspace(0.0, R0MAX, num = 10)
                    r0s[0] = TINY
                    for R0 in r0s:
                        G = R0
                        for c, m in enumerate(['root', 'fsolve']):
                            def test_rP_roots(G=G, R0=R0, KAP=KAP, SIGMA=SIGMA, THETA_V=THETA_V, m=m):
                                phis = np.linspace(0.0, 2.0*np.pi, num = 8)
                                for PHI in phis:
                                    if m == 'fsolve':
                                        RP = fsolve(root_fun, G,
                                                    args = (R0, PHI, KAP, SIGMA, THETA_V),
                                                    fprime=root_jac)[0]
                                    else:
                                        RP = root(root_fun, G,
                                                    args = (R0, PHI, KAP, SIGMA, THETA_V),
                                                    jac = root_jac).x[0]
                                    G = RP
                            # t = timeit.timeit("test_rP_roots()", setup="from __main__ import test_rP_roots")
                            t = np.mean(timeit.Timer("test_rP_roots()", setup="from __main__ import test_rP_roots").repeat(3, 100))
                            # print KAP, THETA_V, R0, m, t
                            ms.append(m)
                            rs.append(R0)
                            ts.append(t)
                            ks.append(KAP)
                            ths.append(THV)
                            cs.append(c)
        print YVAL
        data = [rs, ms, ks, ths, ts, cs]
        df = pd.DataFrame(data)
        df = df.transpose()
        cols = ['R0', 'Method', 'Kappa', 'ThetaV', 'Time', 'C']
        df.columns = cols
        # df = df.round({'R0P': 3})
        g = sns.FacetGrid(df, col='Kappa', row='ThetaV', hue='Method', sharex=False,
                                palette = sns.color_palette("Set1", n_colors=2),
                                legend_out=False) #ylim=(0,2),
        g.map(plt.plot, "R0", "Time", lw = 1)
        # g = sns.factorplot(x="Phi", y="FPhi", hue="C", row="ThetaV", col="Kappa",
                            # data=df, palette = sns.color_palette("Blues", n_colors=len(r0s)))
        g.set_axis_labels(r"$r_0'$", r"Time [s] (Avg. of 3x100 runs)")
        handles, labels = g.fig.get_axes()[0].get_legend_handles_labels()
        g.fig.get_axes()[-2].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # for i in range(len(g.fig.get_axes())):
            # handles, labels = g.fig.get_axes()[i].get_legend_handles_labels()
            # thv_val = float(g.fig.get_axes()[i].get_title().split('|')[0].split('=')[1].strip())
            # kap_val = float(g.fig.get_axes()[i].get_title().split('|')[1].split('=')[1].strip())
            # # print kap_val, thv_val
            # labs = [df.groupby(['Kappa', 'ThetaV', 'C']).get_group((kap_val, thv_val, float(lab)))['Method'].unique()[0] for lab in labels]
            # g.fig.get_axes()[i].legend(handles, labs,
                                                # loc='upper right', 
                                                # bbox_to_anchor=(1.2, 1.0))
        g.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
        plt.suptitle(r"$r'$ {c} ($y={a} | r'_{{0,min}}={b}$)".format(a=YVAL, b=TINY, c="Root Timing Tests"))
        g.fig.subplots_adjust(top=.9)
        # plt.show()
        plt.savefig("rPrime_Root_Timing(y={a}_r0'min={b}).pdf".format(a=YVAL, b=TINY), format="pdf", dpi=1200)

def main():
    order = 6
    TINY = 1.0e-3
    SIGMA = 2.0
    ys = []
    ks = []
    ts = []
    rs = []
    vs = []
    for YVAL in [TINY, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - TINY]:
    # for YVAL in [0.5]:
        for KAP in [0.0, 1.0, 10.0]:
            for THV in [0.0, 1.0, 2.0, 6.0]:
                print YVAL, KAP, THV
                THETA_V = radians(THV)
                R0MAX = r0_max(YVAL, KAP, SIGMA,THETA_V)
                if R0MAX > 0.0:
                    grb = GrbaIntegrator(KAP, THETA_V, SIGMA, 1.0, 0.0, 2.2)
                    r0s = np.linspace(0.0, R0MAX, num = 2**order + 1)
                    # vals = []
                    # vals.append(0.0)
                    for R0 in r0s[1:]:
                        val = grb.r0_integrand(YVAL, R0)
                        val_c = grb.r0_integrand_c(YVAL, R0)
                        err = np.abs(val - val_c) / val_c * 100.0
                        if err > 1.0e-11:
                            print YVAL, KAP, THV, R0, err
                        # vals.append(val)
                    
                    # val = simps(vals, r0s)
                    # print YVAL, KAP, THV, val
                    # r0s[0] = TINY
                    # for R0 in r0s:
                        # val = grb.simps_phi(R0) / np.pi
                        # val = grb.r0_integrand(YVAL, R0)
                        # ys.append(YVAL)
                        # ks.append(KAP)
                        # ts.append(THV)
                        # rs.append(R0)
                        # vs.append(val)
    
    # data = [rs, vs, ks, ts, ys]
    # df = pd.DataFrame(data)
    # df = df.transpose()
    # cols = ['R0', 'PhiInt', 'Kappa', 'ThetaV', 'Y']
    # df.columns = cols
    # df['PhiInt'] = df['PhiInt'].apply(np.log10)
    # df['R0'] = df['R0'].apply(np.log10)
    # g = sns.FacetGrid(df, col='Kappa', row='ThetaV', hue='Y', sharex=False,
                                # palette = sns.color_palette("Set1", n_colors=7),
                                # legend_out=False) #ylim=(0,2),
    # g.map(plt.plot, "R0", "PhiInt", lw = 1)
    # g.set_axis_labels(r"$\log r_0'$", r"$\log (r' / r_0')^2 \times r_0' \times I_{\nu ,G}(y, r_0' | \kappa, \theta_V, \sigma, C_k)$")
    # handles, labels = g.fig.get_axes()[0].get_legend_handles_labels()
    # g.fig.get_axes()[-2].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=7)
    # g.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
    # plt.suptitle(r"$\phi$ Integrand ($r'_{{0,min}}={a}$)".format(a=TINY))
    # g.fig.subplots_adjust(top=.9)
    # g.fig.get_axes()[0].set_ylabel('')
    # g.fig.get_axes()[6].set_ylabel('')
    # g.fig.get_axes()[6].set_xlabel('')
    # g.fig.get_axes()[8].set_xlabel('')
    # # plt.show()
    # plt.savefig("phiIntegrand(r0'min={a}).pdf".format(a=TINY), format="pdf", dpi=1200)

if __name__ == "__main__":
    # test_rP_roots()
    # print np.mean(timeit.Timer("test_rP_roots()", setup="from __main__ import test_rP_roots").repeat(3, 1000))
    sys.exit(int(main() or 0))