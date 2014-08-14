function [ccg, bins] = correlogram(t, assignment, binsize, maxlag)
% Calculate cross-correlograms.
%   ccg = calcCCG(t, assignment, binsize, maxlag) calculates the cross- and
%   autocorrelograms for all pairs of clusters with input
%       t               spike times             #spikes x 1
%       assignment      cluster assignments     #spikes x 1
%       binsize         bin size in ccg         scalar
%       maxlag          maximal lag             scalar
% 
%  and output
%       ccg             computed correlograms   #bins x #clusters x
%                                                               #clusters
%       bins            bin times relative to center    #bins x 1

[t, idx] = sort(t);    % we need the spike times to be sorted
assignment = assignment(idx);
K = max(assignment);
nbins = round(maxlag / binsize);
ccg = zeros(2 * nbins + 1, K, K);
N = numel(t);
j = 1;
for i = 1:N
    while j > 1 && t(j) > t(i) - maxlag
        j = j - 1;
    end
    while j < N && t(j + 1) < t(i) + maxlag
        j = j + 1;
        if i ~= j
            bin = round((t(i) - t(j)) / binsize) + nbins + 1;
            ccg(bin, assignment(i), assignment(j)) = ccg(bin, assignment(i), assignment(j)) + 1;
        end
    end
end
bins = (-nbins : nbins) * binsize;
