/*
 * greedy.c - Greedy search for binary pursuit
 *
 * Performs greedy search for Binary Pursuit algorithm
 */

#include "mex.h"
#include "matrix.h"
#include <math.h>


void printSparse(mxArray *X) 
{
    double *Xpr = mxGetPr(X), *Xpi = mxGetPi(X);
    mwIndex *Xir = mxGetIr(X), *Xjc = mxGetJc(X);
    mwSize M = mxGetM(X), N = mxGetN(X);
    mwSize nz = Xjc[N];
    mexPrintf("X: %d-by-%d (nz = %d, nzmax = %d)\n", M, N, nz, mxGetNzmax(X));
    for (int i = 0; i != nz; ++i) {
        mexPrintf("%d: X[%d] = %.2f + %.2fi\n", i, Xir[i], Xpr[i], Xpi[i]);
    }
    mexPrintf("jc:");
    for (int i = 0; i != N + 1; ++i) {
        mexPrintf(" %d", Xjc[i]);
    }
    mexPrintf("\n\n");
}


int flip(mxArray *X, const mxArray *DL, const mxArray *A, const mxArray *dDL, 
        const mxArray *s, mwSize offset, mwSize T, const mxArray *h, 
        const mxArray *wws, const mxArray *wVs)
{
//     static int count = 0;
//     if (++count > 4096) return -1;
    
    const mwSize *sz = mxGetDimensions(dDL);
    mwSize D = sz[0], M = sz[1], p = sz[3], Ndt = sz[4];
    mwSize N = mxGetM(DL);
    mwSize Tdt = ceil((double) N / (double) Ndt);
    mwSize lenh = mxGetM(h);
    mwSize pad = (lenh - 1) / 2;
    
    // find maximum
    double *DLpr = mxGetPr(DL);
    double max = 0, d;
    int iMax = -1, jMax = -1;
    for (mwSize j = 0; j != M; ++j) {
        for (mwSize i = offset + 1; i != offset + T + 1; ++i) {
            d = DLpr[N * j + i];
            if (d >= max) {
                max = d;
                iMax = i;
                jMax = j;
            }
        }
    }
//     mexPrintf("offset = %d, T = %d, iMax = %d, jMax = %d, max = %.5f\n", offset, T, iMax, jMax, max);
    
    double Xij;
    double *Xpr = mxGetPr(X), *Xpi = mxGetPi(X);
    double *hpr = mxGetPr(h), *Apr = mxGetPr(A);
    mwIndex *Xir = mxGetIr(X), *Xjc = mxGetJc(X);
    double sgn;
    if (max > 0) {

//     mexPrintf("2");
        double a = 0, r = 0;
        int sub;
        mwSize l = Xjc[jMax];
        while (l != Xjc[jMax + 1] && Xir[l] <= iMax) {
            if (Xir[l] == iMax) {
                a = Xpr[l];
                r = Xpi[l];
                break;
            } else {
                ++l;
            }
        }
        
//         printSparse(X);
//         mexPrintf("a = %.1f, r = %.1f, l = %d\n", a, r, l);
//      mexPrintf("l = %d\n", l);
        if (a == 0) {
            // add spike - subsample
            sgn = 1;
            max = 0;
            for (mwSize j = 0; j != p; ++j) {
                double m = 0;
                for (mwSize i = 0; i != lenh; ++i) {
                    m = m + DLpr[jMax * N + iMax - pad + i] * hpr[j * lenh + i];
                }
//      mexPrintf("j = %d, m = %.1f\n", j, m);
                if (m > max) {
                    sub = j;
                    max = m;
                }
            }
//     mexPrintf("4");
            for (mwSize i = 0; i != lenh; ++i) {
                a = a + Apr[N * jMax + iMax - pad + i] * hpr[sub * lenh + i];
            }
            r = (double) (sub - (int) p / 2) / (double) p; // CHECK
            
//      mexPrintf("a = %.1f, r = %.1f, sub = %d, l = %d\n", a, r, sub, l);
            mwSize nzmax = mxGetNzmax(X);
//             mexPrintf("Before growing:\n");
//             printSparse(X);
            if (Xjc[M] == nzmax) { // grow X
//                 mexPrintf("GROWING [size = %d]!!\n", nzmax * sizeof(*Xir));
                nzmax *= 2;
//                 mexPrintf("Old size = %d\n", mxGetNzmax(X));
                mxSetNzmax(X, nzmax);
//                 mexPrintf("New size = %d\n", mxGetNzmax(X));
                Xir = (mwIndex*) mxRealloc(Xir, nzmax * sizeof(*Xir));
                mxSetIr(X, Xir);
                Xpr = (double*) mxRealloc(Xpr, nzmax * sizeof(*Xpr));
                mxSetPr(X, Xpr);
                Xpi = (double*) mxRealloc(Xpi, nzmax * sizeof(*Xpi));
                mxSetPi(X, Xpi);
            }
//             mexPrintf("After growing:\n");
//             printSparse(X);
            
            // real part: amplitude
            // imaginary part: subsample (> 0 => shift right, < 0 => shift left)
            for (mwSize j = jMax; j != M; ++j) {
                ++Xjc[j + 1];
            }
//     mexPrintf("6");
            for (mwSize i = Xjc[M] - 1; i > l; --i) {
                Xir[i] = Xir[i - 1];
                Xpr[i] = Xpr[i - 1];
                Xpi[i] = Xpi[i - 1];
            }
            Xir[l] = iMax;
            Xpr[l] = a;
            Xpi[l] = r;
//             mexPrintf("After update:\n");
//             printSparse(X);
            
//             mexPrintf("JC: ");
//             for (mwSize j = 0; j != M + 1; ++j) {
//                 mexPrintf("%d ", Xjc[j]);
//             }
//             mexPrintf("\n");
//     mexPrintf("7");
                    
        } else {
//     mexPrintf("\n\n\n[[[REMOVING SPIKE]]]\n\nBEFORE:\n");
//     mexPrintf("offset = %d, T = %d, iMax = %d, jMax = %d, max = %.5f\n", offset, T, iMax, jMax, max);
//     mexPrintf("a = %.1f, r = %.1f, l = %d\n", a, r, l);
//     printSparse(X);
//     return -1;
            // remove spike
            sgn = -1;
            for (mwSize j = jMax + 1; j != M + 1; ++j) {
                --Xjc[j];
            }
//     mexPrintf("5b");
            for (; l != Xjc[M]; ++l) {
                Xir[l] = Xir[l + 1];
                Xpr[l] = Xpr[l + 1];
                Xpi[l] = Xpi[l + 1];
            }
//         mexPrintf("AFTER:\n");
//         printSparse(X);
//         return -1;
        }
        
//     mexPrintf("8");
        double DLij = DLpr[N * jMax + iMax]; 
        sub = (int) p / 2 - round(r * (double) p);
        mwSize t = iMax / Tdt;
        
        mwSize start = D * M * (jMax + M * (sub + p * t));
        mwSize ii;
        double dA;
        double *dDLpr = mxGetPr(dDL), *wwspr = mxGetPr(wws), 
               *wVspr = mxGetPr(wVs), *spr = mxGetPr(s);
//     mexPrintf("iMax = %d, jMax = %d, start = %d, D = %d, M = %d, sub = %d, r = %.2f, p = %d, t = %d, a = %.5f\n", iMax, jMax, start, D, M, jMax, sub, r, p, t, a);
        for (mwSize j = 0; j != M; ++j) {
            for (mwSize i = 0; i != D; ++i) {
                dA = dDLpr[start + D * j + i] * a * sgn / wwspr[Ndt * j + t];
                ii = N * j + iMax + spr[i];
                Apr[ii] = Apr[ii] - dA;
                DLpr[ii] = DLpr[ii] - dA * (wVspr[ii] + a * dDLpr[start + D * j + i]);
                DLpr[N * jMax + iMax] = -DLij;
            }
        }
    } else {
        iMax = -1;
    }
//     mexPrintf("0\n");
    
    return iMax;
}


void greedy(mxArray *X, const mxArray *DL, const mxArray *A, const mxArray *dDL, 
            const mxArray *s, mwSize offset, mwSize T, const mxArray *h, 
            const mxArray *wws, const mxArray *wVs)
{
//     mexPrintf("greedy(offset = %d, T = %d, len(X) = %d)\n", offset, T, mxGetJc(X)[mxGetN(X)]);
    static int count = 0;
    mwSize Nmax = 100000;
    int i = 0;
    if (T * mxGetN(X) > Nmax) {
        // divide & conquer: split at current maximum
        i = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
        if (count > 800 && count < 1000) {
            mexPrintf("%d\n", i);
        }
        ++count;
//         if (i == 5185) {
//             double *d = mxGetPr(DL);
//             mwSize N = mxGetM(DL);
//             mexPrintf("dl = [");
//             for (int j = 0; j != 37; ++j) {
//                 mexPrintf("%.5f ", d[N*j+i]);
//             }
//             mexPrintf("]\n");
//             mexPrintf("offset = %d\n", offset);
//         }
        
        if (i >= 0) {
            greedy(X, DL, A, dDL, s, offset, i - offset, h, wws, wVs);
            greedy(X, DL, A, dDL, s, i, T - i + offset, h, wws, wVs);
        }
    } else {
//         mexPrintf("leaving...\n");
//         return;
        // regular loop: greedily searching maximum
        while (i >= 0) {
            i = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
            if (count > 800 && count < 1000) {
                mexPrintf("%d\n", i);
            }
            ++count;
//             mexPrintf("|| %d\n", i);
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *X = prhs[0], *DL = prhs[1], *A = prhs[2], *dDL = prhs[3],
                  *s = prhs[4], *h = prhs[7], *wws = prhs[8], *wVs = prhs[9];
    mwSize offset = mxGetScalar(prhs[5]), T = mxGetScalar(prhs[6]), 
           m = mxGetM(X), n = mxGetN(X);
    if (nlhs > 0) {
        plhs[0] = mxCreateSparse(m, n, 1, mxCOMPLEX);
//         printSparse(plhs[0]);
        greedy(plhs[0], DL, A, dDL, s, offset, T, h, wws, wVs);
    }
}



