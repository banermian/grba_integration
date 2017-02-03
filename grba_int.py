import numpy as np
from scipy.optimize import root, fsolve
from scipy.integrate import quad
from ctypes import cdll, c_double, c_int

class GrbaIntegrator(object):
    def __init__(self, kap, thv, sig, gA, k, p):
        # grbaint = cdll.LoadLibrary("Debug/grba_integration.dll")
        grbaint = cdll.LoadLibrary("Release/grba_integration.dll")
        thetaPrime = grbaint.thetaPrime
        thetaPrime.restype = c_double
        thetaPrime.argtypes = [c_double, c_double, c_double]
        engProf = grbaint.energyProfile
        engProf.restype = c_double
        engProf.argtypes = [c_double, c_double, c_double]
        phiInt = grbaint.phiInt
        phiInt.restype = c_double
        phiInt.argtypes = [c_double, c_double, c_double, c_double]
        fluxG = grbaint.fluxWrap
        fluxG.restype = c_double
        fluxG.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double]
        r0IntDE = grbaint.r0IntDE
        r0IntDE.restype = c_double
        r0IntDE.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double]
        fluxG_ct = grbaint.fluxWrap_ct
        fluxG_ct.restype = c_double
        fluxG_ct.argtypes = (c_int, c_double)
        r0Max = grbaint.r0Max
        r0Max.restype = c_double
        r0Max.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double]
        
        self.kap = kap
        self.thv = thv
        self.sig = sig
        self.gA = gA
        self.k = k
        self.p = p
        self.thetaPrime = thetaPrime
        self.engProf = engProf
        self.fluxG = fluxG
        self.r0IntDE = r0IntDE
        self.fluxG_ct = fluxG_ct
        self.r0Max = r0Max
    
    def _root_fun(self, r, r0, phi, kap, sig, thv):
        thp = self.thetaPrime(r, thv, phi)
        eng = self.engProf(thp, sig, kap)
        lhs = eng*(np.power(r, 2.0) + 2.0*r*np.tan(thv)*np.cos(phi) + np.power(np.tan(thv), 2.0))
        thp0 = self.thetaPrime(r, thv, 0.0)
        eng0 = self.engProf(thp0, sig, kap)
        rhs = np.power(r0 + np.tan(thv), 2.0)*eng0
        return lhs - rhs

    def _root_jac(self, r, r0, phi, kap, sig, thv):
        thp = self.thetaPrime(r, thv, phi)
        first = r + np.tan(thv)*np.cos(phi)
        second = np.power(r, 2.0) + 2.0*r*np.tan(thv)*np.cos(phi) + np.power(np.tan(thv), 2.0)
        frac = (kap*np.log(2.0)*np.power(thp / sig, 2.0*kap)) / (r*(1.0 + 0.5*r*np.sin(2.0*thv)*np.cos(phi)))
        exponent = 2.0*self.engProf(thp, sig, kap)
        return (first - second*frac)*exponent
    
    def simps_phi(self, r0, eps = 1.0e-9):
        NMAX = 25
        sum = 0.0
        osum = 0.0
        for n in xrange(6, NMAX):
            it = 2
            for j in xrange(1, n-1): 
                it <<= 1
            
            tnm = it
            h = 2.0*np.pi / tnm
            s = 2.0
            g = r0
            phi = 0.0
            for i in xrange(1, it):
                phi += h
                rp = root(self._root_fun, g, 
                            args = (r0, phi, self.kap, self.sig, self.thv),
                            jac = self._root_jac).x[0]
                g = rp
                fx = np.power(rp / r0, 2.0)
                
                if (i % 2):
                    s += 4.0*fx
                else:
                    s += 2.0*fx
                    
            sum = s*h / 3.0
            if (np.abs(sum - osum) < eps*np.abs(osum) or (sum == 0.0 and osum == 0.0)):
                return sum
            
            osum = sum
     
    def phi_int(self, r0):
        return self.phiInt(r0, self.kap, self.thv, self.sig)
    
    def _r0_integrand(self, y, r0):
        Gk = (4.0 - self.k)*self.gA**2.0
        thP0 = self.thetaPrime(r0 / y, self.thv, 0.0)
        exp0 = np.power(np.divide(thP0, self.sig), 2.0*self.kap)
        chiVal = np.divide(y - Gk*np.exp2(-exp0)*(np.tan(self.thv) + r0 / y)**2.0, np.power(y, 5.0 - self.k))
        bG = (1.0 - self.p)/2.0
        ys = np.power(y, 0.5*(bG*(4.0 - self.k) + 4.0 - 3.0*self.k))
        chis = np.power(chiVal, np.divide(7.0*self.k - 23.0 + bG*(13.0 + self.k), 6.0*(4.0 - self.k)))
        factor = np.power((7.0 - 2.0*self.k)*chiVal*np.power(y, 4.0 - self.k) + 1.0, bG - 2.0)
        return r0*ys*chis*factor*self.simps_phi(r0 / y)
    
    def _r0_integrand_c(self, y, r0):
        return self.fluxG(y, r0, self.kap, self.sig, self.thv, self.gA, self.k, self.p)
    
    def r0_max(self, y):
        return self.r0Max(y, self.kap, self.sig, self.thv, self.k, self.p, self.gA)
    
    def r0_int(self, y, RMIN):
        return self.r0IntDE(y, RMIN, self.kap, self.sig, self.thv, self.k, self.p, self.gA)
    
    def r0_int_ct(self, y, RMIN, RMAX):
        return quad(self.fluxG_ct, RMIN, RMAX, (y, self.kap, self.sig, self.thv, self.k, self.p, self.gA))[0]

if __name__ == '__main__':
    import timeit
    SIGMA = 2.0
    title = "|   Y   |  KAP   |  THV  |  TIME(s)    |"
    print title
    print "-"*len(title)
    for Y in [0.001, 0.1, 0.5, 0.9, 0.999]:
    # for Y in [0.1]:
        for KAP in [0.0, 1.0, 10.0]:
            for THV in [0.0, 1.0, 3.0]:
                THETA_V = np.radians(THV)
                grb = GrbaIntegrator(KAP, THETA_V, SIGMA, 1.0, 0.0, 2.2)
                intVal = grb.r0_max(Y)
                # intVal = grb.r0_int(Y, 0.00001)
                # s = """from grba_int import *;grb = GrbaIntegrator({}, {}, {}, 1.0, 0.0, 2.2);intVal = grb.r0_int({}, 0.00001)""".format(KAP, THETA_V, SIGMA, Y)
                # intVal = timeit.timeit(stmt = s, number = 10)
                # R0MAX = grb.r0_max(Y)
                # intVal = grb.r0_int_ct(Y, 0.001, R0MAX)
                print "|  {:05.3f}  |  {:04.1f}  |  {}  |  {:05.3f}  |".format(Y, KAP, THV, intVal)