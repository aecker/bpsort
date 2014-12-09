/*
 * correlogram.c - Compute cross-correlograms
 *
 * correlogram(t, a, k, binsize, maxlag) computes the cross-correlograms
 *      of all k neurons. Spike times t are assumed to be sorted and spike
 *      assignments a are [1..k].
 */

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <algorithm>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *t = mxGetPr(prhs[0]);
    double *a = mxGetPr(prhs[1]);
    mwSize N = mxGetM(prhs[0]);
    mwSize K = mxGetScalar(prhs[2]);
    double binsize = mxGetScalar(prhs[3]);
    double maxlag = mxGetScalar(prhs[4]);
    int nbins = round(maxlag / binsize);
    mwSize dims[] = {2 * nbins + 1, K, K};
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double *ccg = mxGetPr(plhs[0]);
    
    mwSize i, j = 0, idx;
    int bin;
    for (i = 0; i != N; ++i) {
        while (j > 0 && t[j] > t[i] - maxlag) {
            j = j - 1;
        }
        while (j < N - 1 && t[j + 1] < t[i] + maxlag) {
            j = j + 1;
            if (i != j) {
                bin = round((double) (t[i] - t[j]) / binsize) + nbins;
                idx = bin + (2 * nbins + 1) * (a[i] - 1 + K * (a[j] - 1));
                ccg[idx] = ccg[idx] + 1;
            }
        }
    }
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(2 * nbins + 1, 1, mxREAL);
        double *bins = mxGetPr(plhs[1]);
        for (i = 0; i != 2 * nbins + 1; ++i) {
            bins[i] = (double) (i - nbins) * binsize;
        }
    }
}
