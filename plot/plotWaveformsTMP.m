function plotWaveforms(w, assignment, varargin)
% Plot waveforms for each cluster.
%   plotWaveforms(w, assignment) plots for all four channels of each
%   cluster 100 sample waveforms, overlaid by the average waveform. All
%   panels are drawn on the same scale to facilitate comparison.

% parse optional inputs
p = inputParser;
p.addOptional('clusters', ':', @(x) isnumeric(x) || ischar(x) && isequal(x, ':'));
p.addOptional('waveforms', 20, @(x) isnumeric(x) && isscalar(x));
p.addOptional('figure', 4, @(x) isnumeric(x) && isscalar(x));
p.parse(varargin{:});
par = p.Results;

colors = hsv(max(assignment));
rng(0);

% keep only selected clusters
if ~ischar(par.clusters)
    keep = ismember(assignment, par.clusters);
    w(:, ~keep, :) = [];
    assignment(~keep) = [];
end
par.clusters = unique(assignment);
K = numel(par.clusters);
colors = colors(par.clusters, :);
C = size(w, 3);

figure(par.figure), clf
bg = 0.3 * ones(1, 3);
set(gcf, 'color', bg)
yl = [0 0];
hdl = zeros(K, C);
for i = 1 : K
    % select random subset of waveforms
    ndx = find(assignment == i);
    rnd = randperm(numel(ndx));
    for j = 1 : C
        hdl(i, j) = axes('Position', [(i - 1) / K, (j - 1) / C, 1 / K, 1 / C], 'color', bg); %#ok<LAXES>
        plot(w(:, ndx(rnd(1 : min(end, par.waveforms))), j), 'k')
        hold on
        plot(mean(w(:, ndx, j), 2), 'color', colors(i, :), 'linewidth', 2)
        axis tight off
        ylj = ylim;
        yl = [min(yl(1), ylj(1)), max(yl(2), ylj(2))];
    end
end

% make axis limit consistent
set(hdl(:), 'ylim', yl);

% plot scale bar
axes(hdl(1, 1))
xl = xlim;
plot(xl(2) * ones(1, 2), yl(1) + [0 100], 'k', 'linewidth', 2)
text(xl(2), yl(1), '100 \muV ', 'horizontalalignment', 'right', 'verticalalignment', 'bottom')

