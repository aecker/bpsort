function plotCCG(t, assignment, varargin)
% Plot cross-correlograms of all pairs.
%   plotCCG(t, assignment) plots a matrix of cross(auto)-correlograms for
%   all pairs of clusters. Inputs are:
%       t           vector of spike times           #spikes x 1
%       assignment  vector of cluster assignments   #spikes x 1

% parse optional inputs
p = inputParser;
p.addOptional('clusters', ':', @(x) isnumeric(x) || ischar(x) && isequal(x, ':'));
p.addOptional('binsize', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addOptional('maxlag', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addOptional('figure', 2, @(x) isnumeric(x) && isscalar(x));
p.parse(varargin{:});
par = p.Results;

% keep only selected clusters
colors = hsv(max(assignment));
if ~ischar(par.clusters)
    keep = ismember(assignment, par.clusters);
    t(~keep) = [];
    assignment(~keep) = [];
end
[par.clusters, ~, assignment] = unique(assignment);
K = numel(par.clusters);
colors = colors(par.clusters, :);

% calculate CCGs
[ccg, bins] = correlogram(t, assignment, par.binsize, par.maxlag);

% plot
figure(par.figure), clf
set(gcf, 'color', 0.5 * ones(1, 3))
for i = 1 : K
    for j = 1 : K
        axes('Position', [j - 1, K - i, 1, 1] / K) 
        hold on
        if i == j
            bar(bins, ccg(:, i, j), 1, 'facecolor', colors(i, :), 'edgecolor', colors(i, :))
        else
            bar(bins, ccg(:, i, j), 1, 'facecolor', 'k')
        end
        axis tight off
        xlim(1.2 * bins([1 end]))
        ylim([0 1.2] .* ylim)
        if i ~= j
            plot(0, 0, '*', 'color', colors(j, :))
        end
    end
end

