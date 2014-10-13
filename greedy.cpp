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
    const mwSize *sz = mxGetDimensions(dDL);
    mwSize D = sz[0], M = sz[1], p = sz[3];
    mwSize Ndt = mxGetNumberOfDimensions(dDL) == 5 ? sz[4] : 1;
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
    
    double Xij;
    double *Xpr = mxGetPr(X), *Xpi = mxGetPi(X);
    double *hpr = mxGetPr(h), *Apr = mxGetPr(A);
    mwIndex *Xir = mxGetIr(X), *Xjc = mxGetJc(X);
    double sgn;
    if (max > 0) {

        // find location in sparse array
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
        
        if (a == 0) { // add spike - subsample
            
            // determine amplitude and subsample shift
            sgn = 1;
            max = 0;
            for (mwSize j = 0; j != p; ++j) {
                double m = 0;
                for (mwSize i = 0; i != lenh; ++i) {
                    m = m + DLpr[jMax * N + iMax - pad + i] * hpr[j * lenh + i];
                }
                if (m > max) {
                    sub = j;
                    max = m;
                }
            }
            for (mwSize i = 0; i != lenh; ++i) {
                a = a + Apr[N * jMax + iMax - pad + i] * hpr[sub * lenh + i];
            }
            r = (double) (sub - (int) p / 2) / (double) p; // CHECK

            // grow sparse array if necessary
            mwSize nzmax = mxGetNzmax(X);
            if (Xjc[M] == nzmax) { // grow X
                nzmax *= 2;
                mxSetNzmax(X, nzmax);
                Xir = (mwIndex*) mxRealloc(Xir, nzmax * sizeof(*Xir));
                mxSetIr(X, Xir);
                Xpr = (double*) mxRealloc(Xpr, nzmax * sizeof(*Xpr));
                mxSetPr(X, Xpr);
                Xpi = (double*) mxRealloc(Xpi, nzmax * sizeof(*Xpi));
                mxSetPi(X, Xpi);
            }
            
            // add values to sparse array
            //   real: amplitude
            //   imag: subsample (> 0 => shift right, < 0 => shift left)
            for (mwSize j = jMax; j != M; ++j) {
                ++Xjc[j + 1];
            }
            for (mwSize i = Xjc[M] - 1; i > l; --i) {
                Xir[i] = Xir[i - 1];
                Xpr[i] = Xpr[i - 1];
                Xpi[i] = Xpi[i - 1];
            }
            Xir[l] = iMax;
            Xpr[l] = a;
            Xpi[l] = r;
                    
        } else { // remove spike
            
            sgn = -1;
            for (mwSize j = jMax + 1; j != M + 1; ++j) {
                --Xjc[j];
            }
            for (; l != Xjc[M]; ++l) {
                Xir[l] = Xir[l + 1];
                Xpr[l] = Xpr[l + 1];
                Xpi[l] = Xpi[l + 1];
            }
        }

        // update change in posterior
        double DLij = DLpr[N * jMax + iMax]; 
        sub = (int) p / 2 - round(r * (double) p);
        mwSize t = iMax / Tdt;
        mwSize start = D * M * (jMax + M * (sub + p * t));
        mwSize ii;
        double dA;
        double *dDLpr = mxGetPr(dDL), *wwspr = mxGetPr(wws), 
               *wVspr = mxGetPr(wVs), *spr = mxGetPr(s);
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
    return iMax;
}


void greedy(mxArray *X, const mxArray *DL, const mxArray *A, const mxArray *dDL, 
            const mxArray *s, mwSize offset, mwSize T, const mxArray *h, 
            const mxArray *wws, const mxArray *wVs)
{
    mwSize Nmax = 100000;
    int i = 0;
    if (T * mxGetN(X) > Nmax) {
        // divide & conquer: split at current maximum
        i = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
        if (i >= 0) {
            greedy(X, DL, A, dDL, s, offset, i - offset, h, wws, wVs);
            greedy(X, DL, A, dDL, s, i, T - i + offset, h, wws, wVs);
        }
    } else {
        while (i >= 0) {
            i = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
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
        greedy(plhs[0], DL, A, dDL, s, offset, T, h, wws, wVs);
    }
}
