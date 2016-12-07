import numpy as np
from scipy.optimize import root, fsolve
from ctypes import cdll, c_double

class GrbaIntegrator(object):
    def __init__(self, kap, thv, sig, gL, k, p):
        grbaint = cdll.LoadLibrary("Release/grba_integration.dll")
        thetaPrime = grbaint.thetaPrime
        thetaPrime.restype = c_double
        thetaPrime.argtypes = [c_double, c_double, c_double]
        engProf = grbaint.energyProfile
        engProf.restype = c_double
        engProf.argtypes = [c_double, c_double, c_double]
        
        self.kap = kap
        self.thv = thv
        self.sig = sig
        self.gL = gL
        self.k = k
        self.p = p
        self.thetaPrime = thetaPrime
        self.engProf = engProf
        
    
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
     
    def r0_integrand(self, y, r0):
        Gk = (4.0 - self.k)*self.gL**2.0
        thP0 = self.thetaPrime(r0, self.thv, 0.0)
        exp0 = np.power(np.divide(thP0, self.sig), 2.0*self.kap)
        chiVal = np.divide(y - Gk*np.exp2(-exp0)*(np.tan(self.thv) + r0)**2.0, np.power(y, 5.0 - self.k))
        bG = (1.0 - self.p)/2.0
        ys = np.power(y, 0.5*(bG*(4.0 - self.k) + 4.0 - 3.0*self.k))
        chis = np.power(chiVal, np.divide(7.0*self.k - 23.0 + bG*(13.0 + self.k), 6.0*(4.0 - self.k)))
        factor = np.power((7.0 - 2.0*self.k)*chiVal*np.power(y, 4.0 - self.k) + 1.0, bG - 2.0)
        return r0*ys*chis*factor*self.simps_phi(r0)