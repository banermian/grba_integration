#define _USE_MATH_DEFINES
#define DLLEXPORT extern "C" __declspec(dllexport)
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include "cminpack.h"

const double TORAD = M_PI / 180.0;

struct params;
DLLEXPORT double thetaPrime(double r, double thv, double phi);
DLLEXPORT double energyProfile(double thp, double sig, double kap);
class RootFuncPhi;
//struct RootFuncPhi;
//double rtnewtPhi(RootFuncPhi& func, const double g, const double xacc);
//double rtsafePhi(RootFuncPhi *func, const double x1, const double x2, const double xacc);
//void testRootSolve();
int fcn(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag);
double rootPhi(RootFuncPhi& func, const double g, const double xacc);
double simpsPhi(params& ps, const double r0, const double a, const double b, const double eps = 1.0e-9);
DLLEXPORT double phiInt(const double r0, const double kap, const double thv, const double sig);
double intG(double y, double chi, const double k, const double p);
DLLEXPORT double fluxG(params& ps, const double y, const double r0);
DLLEXPORT double fluxWrap(double y, double r0, const double kap, const double sig, const double thv, const double gA, const double k, const double p);
void testPhiInt();
double milneR0(params& ps, const double y, const double a, const double b, const double eps = 1.0e-7);
DLLEXPORT double r0Int(double y, double r0Min, double r0Max, const double kap, const double sig, const double thv, const double gA, const double k, const double p);
void testR0Int();
struct RootFuncR0;
double rtsafeR0(RootFuncR0& func, const double x1, const double x2, const double xacc);

int main(void)
{
    //testRootSolve();
    testPhiInt();
    // double result = phiInt(0.1, 1.0, 6.0, 2.0);
    // std::cout << result << std::endl;
    //testR0Int();

    return 0;
}

struct params {
    const double KAP;
    const double SIG;
    const double THV;
    const double K;
    const double P;
    const double GA;
};

DLLEXPORT double thetaPrime(double r, double thv, double phi) {
    double numer = r*pow(pow(cos(thv), 2) - 0.25*pow(sin(2.0*thv), 2)*pow(cos(phi), 2), 0.5);
    double denom = 1.0 + 0.5*r*sin(2.0*thv)*cos(phi);
    return numer / denom;
}

DLLEXPORT double energyProfile(double thp, double sig, double kap) {
    return exp2(-pow(thp / sig, 2.0*kap));
}

//struct RootFuncPhi
//{
//    const double r0, kap, sig, thv;
//    double phi;
//    RootFuncPhi(double PHI, const double R0, params *p) :
//        phi(PHI), r0(R0), kap(p->KAP), sig(p->SIG), thv(p->THV) {}
//    double f(double r) {
//        double thp = thetaPrime(r, thv, phi);
//        double eng = energyProfile(thp, sig, kap);
//        double lhs = (pow(r, 2) + 2.0*r*tan(thv)*cos(phi) + pow(tan(thv), 2))*eng;
//        double thp0 = thetaPrime(r, thv, 0.0);
//        double eng0 = energyProfile(thp0, sig, kap);
//        double rhs = pow(r0 + tan(thv), 2)*eng0;
//        return lhs - rhs;
//    }
//    double df(double r) {
//        double thp = thetaPrime(r, thv, phi);
//        double first = r + tan(thv)*cos(phi);
//        double second = pow(r, 2) + 2.0*r*tan(thv)*cos(phi) + pow(tan(thv), 2);
//        double frac = (kap*log(2.0)*pow(thp / sig, 2.0*kap)) / (r*(1.0 + 0.5*r*sin(2.0*thv)*cos(phi)));
//        double exponent = 2.0*energyProfile(thp, sig, kap);
//        return (first - second*frac)*exponent;
//    }
//};

class RootFuncPhi
{
public:
    RootFuncPhi(const double PHI, const double R0, params &PS) :
        phi(PHI), r0(R0), kap(PS.KAP), sig(PS.SIG), thv(PS.THV)
    {}

    double f(double r) {
        double thp = thetaPrime(r, thv, phi);
        double eng = energyProfile(thp, sig, kap);
        double lhs = (pow(r, 2) + 2.0*r*tan(thv)*cos(phi) + pow(tan(thv), 2))*eng;
        double thp0 = thetaPrime(r, thv, 0.0);
        double eng0 = energyProfile(thp0, sig, kap);
        double rhs = pow(r0 + tan(thv), 2)*eng0;
        return lhs - rhs;
    }

    double df(double r) {
        double thp = thetaPrime(r, thv, phi);
        double first = r + tan(thv)*cos(phi);
        double second = pow(r, 2) + 2.0*r*tan(thv)*cos(phi) + pow(tan(thv), 2);
        double frac = (kap*log(2.0)*pow(thp / sig, 2.0*kap)) / (r*(1.0 + 0.5*r*sin(2.0*thv)*cos(phi)));
        double exponent = 2.0*energyProfile(thp, sig, kap);
        return std::abs((first - second*frac)*exponent);
    }

private:
    const double phi, r0, kap, sig, thv;
};

int fcn(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
{
    /*      subroutine fcn for hybrj example. */
    (void)p;

    if (iflag != 2)
    {
        fvec[0] = ((RootFuncPhi*)p)->f(x[0]);
    }
    else
    {
        fjac[0] = ((RootFuncPhi*)p)->df(x[0]);
    }
    return 0;
}

double rootPhi(RootFuncPhi& func, double g, const double xacc) {
    int n, ldfjac, info, lwa;
    double tol, fnorm;
    double x[1], fvec[1], fjac[1 * 1], wa[99];

    n = 1;
    ldfjac = 1;
    lwa = 99;

    //tol = sqrt(__cminpack_func__(dpmpar)(1));
    tol = xacc;

    x[0] = g;

    void *p = NULL;
    p = &func;

    info = __cminpack_func__(hybrj1)(fcn, p, n, x, fvec, fjac, ldfjac, tol, wa, lwa);

    //fnorm = __cminpack_func__(enorm)(n, fvec);

    return (double)x[0];
};

//double rtnewtPhi(RootFuncPhi& func, const double g, const double xacc) {
//    const int JMAX = 20;
//    int j;
//    double root = g;
//    for (j = 0; j < JMAX; j++) {
//        double f = func.f(root);
//        double df = func.df(root);
//        double dx = f / df;
//        root -= dx;
//        if (std::abs(dx) < xacc) {
//            return root;
//        }
//    }
//    std::cout << "rtnewtPhi did not converge. Ended with j = " << j << ", g = " << g << " , and root = " << root << std::endl;
//    throw("Maximum number of iterations exceeded in rtnewt");
//}
//
//double rtsafePhi(RootFuncPhi *func, const double x1, const double x2, const double xacc) {
//    const int MAXIT = 100;
//    double xl, xh;
//    double fl = func->f(x1);
//    double fh = func->f(x2);
//    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
//        std::cout << "Root must be bracketed in rtsafePhi" << std::endl;
//        printf_s("xl = %f, fl = %f, xh = %f, fh = %f\n", x1, fl, x2, fh);
//        throw("Root must be bracketed in rtsafePhi");
//    }
//    if (fl == 0.0) return x1;
//    if (fh == 0.0) return x2;
//    if (fl < 0.0) {
//        xl = x1;
//        xh = x2;
//    }
//    else {
//        xh = x1;
//        xl = x2;
//    }
//    double rts = 0.5*(x1 + x2);
//    double dxold = std::abs(x2 - x1);
//    double dx = dxold;
//    double f = func->f(rts);
//    double df = func->df(rts);
//    int j;
//    for (j = 0; j < MAXIT; j++) {
//        if ((((rts - xh)*df - f)*((rts - xl)*df - f) > 0.0)
//            || (std::abs(2.0*f) > std::abs(dxold*df))) {
//            dxold = dx;
//            dx = 0.5*(xh - xl);
//            rts = xl + dx;
//            if (xl == rts) return rts;
//        }
//        else {
//            dxold = dx;
//            dx = f / df;
//            double temp = rts;
//            rts -= dx;
//            if (temp == rts) return rts;
//        }
//        if (std::abs(dx) < xacc) return rts;
//        f = func->f(rts);
//        df = func->df(rts);
//        if (f < 0.0)
//            xl = rts;
//        else
//            xh = rts;
//    }
//    std::cout << "rtsafePhi did not converge. Ended with j = " << j << " , and root = " << rts << std::endl;
//    throw("Maximum number of iterations exceeded in rtsafePhi");
//}
//
//void testRootSolve() {
//    double G;
//    int NSTEPS;
//    double R0;
//
//    std::cout << "Enter r0 value: ";
//    std::cin >> R0;
//
//    G = R0;
//    params PARAMS = { 1.0, 2.0, 6.0*TORAD, 0.0, 2.2, 1.0 };
//
//    std::cout << "Enter desired number of steps: ";
//    std::cin >> NSTEPS;
//
//    FILE *ofile;
//    errno_t err;
//    char filename[50];
//    sprintf_s(filename, 50, "phiRoot_r0=%f_kap=%f_thv=%f.txt", R0, PARAMS.KAP, PARAMS.THV);
//    err = fopen_s(&ofile, filename, "w");
//    fprintf(ofile, "PHI\tR\tNSTEPS\n");
//    for (int p = 0; p <= NSTEPS; p++) {
//        double phi = 360.0*TORAD*p / NSTEPS;
//        RootFuncPhi rfunc(phi, R0, &PARAMS);
//        //std::pair <double, int> root = rtnewtPhi(&rfunc, G, 1.0e-11);
//        //fprintf(ofile, "%3.2f\t%f\t%d\n", phi, root.first, root.second);
//        //G = root.first;
//        //std::cout << G << std::endl;
//    }
//    fclose(ofile);
//}

//double simpsPhi(params *ps, const double r0, const double a, const double b, const double eps) {
//    const int NMAX = 25;
//    double sum, osum = 0.0;
//    for (int n = 6; n < NMAX; n++) {
//        int it, j;
//        double h, s, x, g, tnm;
//        for (it = 2, j = 1; j<n - 1; j++) it <<= 1;
//        tnm = it;
//        h = (b - a) / tnm;
//        s = 2.0;
//        g = r0;
//        x = a;
//        for (int i = 1; i < it; i++, x += h) {
//            RootFuncPhi rfunc(x, r0, ps);
//            double rp = rtnewtPhi(rfunc, g, 1.0e-9);
//            //double rp = rtsafePhi(&rfunc, 0.9*g, 10.0*g, 1.0e-9);
//            g = rp;
//            double fx = pow(rp / r0, 2.0);
//            if (i % 2) {
//                s += 4.0*fx;
//            }
//            else {
//                s += 2.0*fx;
//            }
//        }
//
//        sum = s*h / 3.0;
//        if (n > 3)
//            if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
//                //std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
//                return sum;
//            }
//        osum = sum;
//    }
//    throw("Maximum number of iterations exceeded in simpsPhi");
//}

double simpsPhi(params& ps, const double r0, const double a, const double b, const double eps) {
    const int NMAX = 25;
    double sum, osum = 0.0;
    for (int n = 6; n < NMAX; n++) {
        int it, j;
        double h, s, x, g, tnm;
        for (it = 2, j = 1; j<n - 1; j++) it <<= 1;
        tnm = it;
        h = (b - a) / tnm;
        s = 2.0;
        g = r0;
        x = a;
        for (int i = 1; i < it; i++, x += h) {
            RootFuncPhi rfunc(x, r0, ps);
            double rp = rootPhi(rfunc, g, 1.0e-9);
            g = rp;
            double fx = pow(rp / r0, 2.0);
            if (i % 2) {
                s += 4.0*fx;
            }
            else {
                s += 2.0*fx;
            }
        }

        sum = s*h / 3.0;
        if (n > 3)
            if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
                //std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
                return sum;
            }
        osum = sum;
    }
    throw("Maximum number of iterations exceeded in simpsPhi");
}

DLLEXPORT double phiInt(const double r0, const double kap, const double thv, const double sig) {
    params PS = { kap, sig, thv, 0.0, 2.2, 1.0 };
    double sumVal = simpsPhi(PS, r0, 0.0, 2.0*M_PI);
    return sumVal;
}

double intG(double y, double chi, const double k, const double p) {
    const double bG = (1.0 - p) / 2.0;
    double ys = pow(y, 0.5*(bG*(4.0 - k) + 4.0 - 3.0*k));
    double chis = pow(chi, (7.0*k - 23.0 + bG*(13.0 + k)) / (6.0*(4.0 - k)));
    double fac = pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bG - 2.0);
    return ys*chis*fac;
}

DLLEXPORT double fluxG(params& ps, const double y, double r0) {
    const double kap = ps.KAP;
    const double sig = ps.SIG;
    const double thv = ps.THV;
    const double gA = ps.GA;
    const double k = ps.K;
    const double p = ps.P;

    const double Gk = (4.0 - k)*pow(gA, 2.0);
    double thP0 = thetaPrime(r0, thv, 0.0);
    double exp0 = pow(thP0 / sig, 2.0*kap);
    double chiVal = (y - Gk*exp2(-exp0)*pow(tan(thv) + r0, 2.0)) / (pow(y, 5.0 - k));
    return r0*intG(y, chiVal, k, p)*simpsPhi(ps, r0, 0.0, 2.0*M_PI);
}

DLLEXPORT double fluxWrap(double y, double r0, const double kap, const double sig, const double thv, const double gA, const double k, const double p) {
    params PS = { kap, sig, thv, k, p, gA };
    double fluxVal = fluxG(PS, y, r0);
    return fluxVal;
}

double milneR0(params& ps, const double y, const double a, const double b, const double eps) {
    const int NMAX = 25;
    double sum, osum = 0.0;
    for (int n = 2; n < NMAX; n++) {
        int it, j;
        double h, s, x1, x2, x3, tnm;
        double f1, f2, f3;
        for (it = 4, j = 1; j < n - 1; j++) it <<= 1;
        tnm = it;
        h = (b - a) / tnm;
        s = 0.0;
        //g = y;
        // x = a + h;
        for (int i = 1; i <= it / 4; i++) {
            x1 = a + (4 * i - 3)*h;
            x2 = a + (4 * i - 2)*h;
            x3 = a + (4 * i - 1)*h;

            //printf_s("x1 = %f, x2 = %f, x3 = %f\n", x1, x2, x3);

            f1 = fluxG(ps, y, x1);
            f2 = fluxG(ps, y, x2);
            f3 = fluxG(ps, y, x3);

            //printf_s("f1 = %e, f2 = %e, f3 = %e\n", f1, f2, f3);

            s = s + 2.0*f1 - f2 + 2.0*f3;
            //std::cout << "s = " << s << std::endl;

            //RootFuncPhi rfunc1(x1, y, ps);
            //RootFuncPhi rfunc2(x2, y, ps);
            //RootFuncPhi rfunc3(x3, y, ps);
            //double rp1 = rtnewtPhi(&rfunc1, g, 1.0e-10);
            //g = rp1;
            //double rp2 = rtnewtPhi(&rfunc2, g, 1.0e-10);
            //g = rp2;
            //double rp3 = rtnewtPhi(&rfunc3, g, 1.0e-10);
            //g = rp3;
            //s = s + 2.0*pow(rp1 / y, 2.0) - pow(rp2 / y, 2.0) + 2.0*pow(rp3 / y, 2.0);
        }

        sum = s*h * 4 / 3;
        //std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
        if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
            std::cout << "milneR0 converged with " << it << " sub-intervals." << std::endl;
            return sum;
        }
        osum = sum;
    }
    std::cout << "milneR0 did not converge. Ended with sum = " << sum << std::endl;
    throw("Maximum number of iterations exceeded in milneR0");
}

DLLEXPORT double r0Int(double y, double r0Min, double r0Max, const double kap, const double sig, const double thv, const double gA, const double k, const double p) {
    params PS = { kap, sig, thv, k, p, gA };
    double r0IntVal = milneR0(PS, y, r0Min, r0Max);
    return r0IntVal;
}

void testR0Int() {
    double Y, KAP, THV;
    std::cout << "Enter values (Y, KAP, THV): " << std::endl;
    std::cin >> Y >> KAP >> THV;
    params PS = { KAP, 2.0, THV*TORAD, 0.0, 2.2, 1.0 };

    //double phiSum = simpsPhi(&PS, R0, 0.0, 2.0*M_PI);
    double r0Sum = milneR0(PS, Y, 0.001, 0.1);
    std::cout << r0Sum << std::endl;
    //std::cout << phiSum << "\t" << r0Sum << "\t" << std::abs(r0Sum - phiSum) / phiSum * 100.0 << std::endl;
}

struct RootFuncR0
{
    const double y, kap, sig, thv, k, gA;
    RootFuncR0(const double Y, params& p) :
        y(Y), kap(p.KAP), sig(p.SIG), thv(p.THV), k(p.K), gA(p.GA) {}
    double f(double r0) {
        const double Gk = (4.0 - k)*pow(gA, 2.0);
        double thp0 = thetaPrime(r0, thv, 0.0);
        double eng0 = energyProfile(thp0, sig, kap);
        double lhs = pow(r0 + tan(thv), 2)*eng0;
        double rhs = (y - pow(y, 5.0 - k)) / Gk;
        return lhs - rhs;
    }
    double df(double r0) {
        double thp0 = thetaPrime(r0, thv, 0.0);
        double frac = kap*log(2.0)*pow(thp0 / sig, 2.0*kap)*((r0 + tan(thv)) / (r0*(1.0 + r0*sin(thv)*cos(thv))));
        double exponent = 2.0*energyProfile(thp0, sig, kap);
        return (1.0 - frac)*exponent;
    }
};

double rtsafeR0(RootFuncR0& func, const double x1, const double x2, const double xacc) {
    const int MAXIT = 100;
    double xl, xh;
    double fl = func.f(x1);
    double fh = func.f(x2);
    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
        std::cout << "Root not bracketed in rtsafeR0" << std::endl;
        std::cout << "fl = " << fl << ", fh = " << fh << std::endl;
        return -1.0;
        //throw("Root must be bracketed in rtsafeR0");
    };
    if (fl == 0.0) return x1;
    if (fh == 0.0) return x2;
    if (fl < 0.0) {
        xl = x1;
        xh = x2;
    }
    else {
        xh = x1;
        xl = x2;
    }
    double rts = 0.5*(x1 + x2);
    double dxold = std::abs(x2 - x1);
    double dx = dxold;
    double f = func.f(rts);
    double df = func.df(rts);
    int j;
    for (j = 0; j < MAXIT; j++) {
        if ((((rts - xh)*df - f)*((rts - xl)*df - f) > 0.0)
            || (std::abs(2.0*f) > std::abs(dxold*df))) {
            dxold = dx;
            dx = 0.5*(xh - xl);
            rts = xl + dx;
            if (xl == rts) return rts;
        }
        else {
            dxold = dx;
            dx = f / df;
            double temp = rts;
            rts -= dx;
            if (temp == rts) return rts;
        }
        if (std::abs(dx) < xacc) return rts;
        f = func.f(rts);
        df = func.df(rts);
        if (f < 0.0)
            xl = rts;
        else
            xh = rts;
    }
    std::cout << "rtsafeR0 did not converge. Ended with j = " << j << " , and root = " << rts << std::endl;
    throw("Maximum number of iterations exceeded in rtsafeR0");
}

void testPhiInt() {
    double YVAL;
    std::cout << "Enter y value: " << std::endl;
    std::cin >> YVAL;
    std::vector<double> KAPPAS = { 0.0, 1.0, 10.0 };
    std::vector<double> THETAS = { 0.0, 1.0, 2.0, 6.0 };
    for (double KAP : KAPPAS) {
        for (double THV : THETAS) {
            params PS = { KAP, 2.0, THV*TORAD, 0.0, 2.2, 1.0 };
            RootFuncR0 r0func(YVAL, PS);
            double R0MAX = rtsafeR0(r0func, 0.0, 0.6, 1.0e-7);
            printf("Kappa = %02.1f, Theta_V = %02.1f, R0'_max = %03.5f \n", KAP, THV, R0MAX);
            double intVal;
            if (R0MAX < 0.0)
            {
                printf("R0'_max negative. \n");
                intVal = 0.0;
            }
            else
            {
                intVal = milneR0(PS, YVAL, 0.001, R0MAX, 1.0e-5);
                printf_s("R0' Integral = %03.5f\n", intVal);
                /*double step = R0MAX / 8;
                printf_s("%f\t%f\t%f\t%f\n", KAP, THV, R0MAX, step);
                for (int i = 1; i < 9; i++) {
                    double R0 = i*step;
                    double sumVal = simpsPhi(PS, R0, 0.0, 2.0*M_PI);
                    printf_s("%d\t%f\t%f\n", i, R0, sumVal);
                }*/
            }
        }
    }
    //params PS = { KAP, THV*TORAD, 2.0, 1.0, 0.0, 2.2 };
    /*FILE * ofile;
    errno_t err;
    char filename[50];
    sprintf_s(filename, 50, "phiInt_r0=%f_kap=%f_thv=%f.csv", R0, PS.KAP, PS.THV);
    err = fopen_s(&ofile, filename, "w");
    fprintf(ofile, "Step,Phi,f(phi)\n");*/
    //double sumVal = simpsPhi(&PS, R0, 0.0, 2.0*M_PI, ofile);
    //double sumVal = simpsPhi(&PS, R0, 0.0, 2.0*M_PI);
    //fclose(ofile);
    /*std::cout << sumVal << std::endl;*/
}