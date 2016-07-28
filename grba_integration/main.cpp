#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

const double TORAD = M_PI / 180.0;

struct params;
struct params {
    const double KAP;
    const double THV;
    const double SIG;
    const double K;
    const double P;
};
double thetaPrime(double r, double thv, double phi);
double energyProfile(double thp, double sig, double kap);
struct RootFuncPhi;
void testRootSolve();
std::pair <double, int> rtnewtPhi(RootFuncPhi *func, const double g, const double xacc);
std::pair <double, int> rtnewtAlpha(RootFuncPhi *func, const double g, const double xacc);
struct Quadrature;
struct TrapzdPhi;
void testPhiInt();
double simpsPhi(params *ps, const double r0, const double a, const double b, const double eps = 1.0e-9);

int main(void)
{
    //void testRootSolve();
    testPhiInt();
    //params PS = { 0.0, 0.0*TORAD, 2.0, 0.0, 2.2 };
    //const int NMAX = 20;
    //double sum, osum, eps = 1.0e-7;
    //int it, j;
    //for (int n = 0; n < NMAX; n++) {
    //    for (it = 1, j = 1; j<n - 1; j++) it <<= 1;
    //    sum = simpsPhi(&PS, 0.1, 0.0, 2.0*M_PI, it);
    //    std::cout << "Num Steps = " << it << ",\tSum = " << sum << std::endl;
    //    if (n > 5)
    //        if (std::abs(sum - osum) < eps*std::abs(osum) ||
    //            (sum == 0.0 && osum == 0.0)) {
    //            std::cout << "sum found" << std::endl;
    //            //return sum;
    //            break;
    //        }
    //    osum = sum;
    //}
    
    return 0;
}

double thetaPrime(double r, double thv, double phi) {
    double numer = r*pow(pow(cos(thv), 2) - 0.25*pow(sin(2.0*thv), 2)*pow(cos(phi), 2), 0.5);
    double denom = 1.0 + 0.5*r*sin(2.0*thv)*cos(phi);
    return numer / denom;
}

double energyProfile(double thp, double sig, double kap) {
    return exp2(-pow(thp / sig, 2.0*kap));
}

struct RootFuncPhi
{
    const double r0, kap, sig, thv;
    double phi;
    RootFuncPhi(double PHI, const double R0, params *p) :
        phi(PHI), r0(R0), kap(p->KAP), sig(p->SIG), thv(p->THV) {}
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
        return (first - second*frac)*exponent;
    }
};

//template <class T>
std::pair <double, int> rtnewtPhi(RootFuncPhi *func, const double g, const double xacc) {
    //double rtnewt(T &func, const double g, const double xacc) {
    const int JMAX = 20;
    double root = g;
    for (int j = 0; j < JMAX; j++) {
        double f = func->f(root);
        double df = func->df(root);
        double dx = f / df;
        root -= dx;
        //printf_s("j=%d\tdx=%e\troot=%f\n", j, dx, root);
        if (std::abs(dx) < xacc) {
            std::pair <double, int> rtnPair(root, j + 1);
            return rtnPair;
            //return root;
        }
    }
    throw("Maximum number of iterations exceeded in rtnewt");
}

//template <class T>
//std::pair <double, int> rtnewtAlpha(T &func, const double g, const double xacc) {
std::pair <double, int> rtnewtAlpha(RootFuncPhi *func, const double g, const double xacc) {
    //double rtnewt(T &func, const double g, const double xacc) {
    const int JMAX = 25;
    double root = g, alpha = 1.0, oldAlpha, resBef, resAft;
    for (int j = 0; j < JMAX; j++) {
        double f = func->f(root);
        double df = func->df(root);
        double dx = f / df;
        root -= alpha*dx;
        oldAlpha = alpha;
        resBef = f;
        resAft = func->f(root);
        printf_s("j=%d\talpha=%f\tresBef=%f\tresAft=%f\troot=%f\tdx=%e\n", j, alpha, resBef, resAft, root, dx);
        if (std::abs(dx) < xacc) {
            std::pair <double, int> rtnPair(root, j);
            return rtnPair;
            //return root;
        }
        else if (std::abs(resAft) > std::abs(resBef)) {
            alpha *= 0.5;
            resAft = resBef;
            root += oldAlpha*dx;
        }
        else {
            alpha *= 1.5;
        }
    }
    throw("Maximum number of iterations exceeded in rtnewt");
}

struct Quadrature {
    int n;
    virtual double next() = 0;
};

//template<class T>
struct TrapzdPhi : Quadrature {
    const double a, b, r0;
    double s = 0, g;
    params *p;
    //T &func;
    //TrapzdPhi() {};
    /*TrapzdPhi(T &funcc, const double aa, const double bb) :
    func(funcc), a(aa), b(bb) {
    n = 1;
    }*/
    TrapzdPhi(params *pp, const double R0, const double aa, const double bb) :
        p(pp), r0(R0), a(aa), b(bb) {
        n = 1;
    }
    double next() {
        double x, tnm, sum, del;
        int it, j;
        n++;

        for (it = 1, j = 1; j < n - 1; j++) it <<= 1;
        tnm = it;
        del = (b - a) / tnm;
        x = 0.0;
        g = r0;
        for (sum = 0.0, j = 0; j <= it; j++, x += del) {
            //sum += func(x);
            RootFuncPhi rfunc(x, r0, p);
            std::pair <double, int> rPair = rtnewtPhi(&rfunc, g, 1.0e-11);
            double rp = rPair.first;
            g = rp;
            sum += pow(rp / r0, 2.0);
        }
        s = 0.5*(s + (b - a)*sum / tnm);
        printf_s("n=%d\tsum=%e\ts=%e\n", n, sum, s);
        return s;
    }
};

double simpsPhi(params *ps, const double r0, const double a, const double b, const double eps) {
    const int NMAX = 25;
    double sum, osum = 0.0;
    for (int n = 2; n < NMAX; n++) {
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
            std::pair <double, int> rPair = rtnewtPhi(&rfunc, g, 1.0e-11);
            double rp = rPair.first;
            g = rp;
            if (i % 2) {
                s += 4.0*pow(rp / r0, 2.0);
            }
            else {
                s += 2.0*pow(rp / r0, 2.0);
            }
        }

        sum = s*h / 3;
        if (n > 3)
            if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
                std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
                return sum;
            }
        osum = sum;
        //std::cout << i << std::endl;
    }
}

void testRootSolve() {
    double G;
    int NSTEPS;
    double R0;

    std::cout << "Enter r0 value: ";
    std::cin >> R0;

    G = R0;
    params PARAMS = { 1.0, 6.0*TORAD, 2.0, 0.0, 2.2 };

    std::cout << "Enter desired number of steps: ";
    std::cin >> NSTEPS;

    FILE * ofile;
    errno_t err;
    char filename[50];
    sprintf_s(filename, 50, "phiRoot_r0=%f_kap=%f_thv=%f.txt", R0, PARAMS.KAP, PARAMS.THV);
    err = fopen_s(&ofile, filename, "w");
    fprintf(ofile, "PHI\tR\tNSTEPS\n");
    for (int p = 0; p <= NSTEPS; p++) {
        double phi = 360.0*TORAD*p / NSTEPS;
        RootFuncPhi rfunc(phi, R0, &PARAMS);
        std::pair <double, int> root = rtnewtPhi(&rfunc, G, 1.0e-11);
        fprintf(ofile, "%3.2f\t%f\t%d\n", phi, root.first, root.second);
        G = root.first;
        std::cout << G << std::endl;
    }
    fclose(ofile);
}

//template <class T>
void testPhiInt() {
    double R0, KAP, THV;
    std::cout << "Enter values (R0, KAP, THV): " << std::endl;
    std::cin >> R0 >> KAP >> THV;
    params PS = { KAP, THV*TORAD, 2.0, 0.0, 2.2 };
    double sumVal = simpsPhi(&PS, R0, 0.0, 2.0*M_PI);
}