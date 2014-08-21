function plotSeparation(b, mu, Sigma, priors, assignment, varargin)
% Plot cluster separation by projecting on LDA axes
%   plotSeparation(b, mu, Sigma, priors, assignment) visualizes the cluster
%   separation by projecting the data on the LDA axis for each pair of
%   clusters. Each column is normalized such that the left (i.e. first)
%   cluster has zero mean and unit variances. The LDA axis is estimated
%   from the model.

% parse optional inputs
p = inputParser;
p.addOptional('clusters', ':', @(x) isnumeric(x) || ischar(x) && isequal(x, ':'));
p.addOptional('figure', 3, @(x) isnumeric(x) && isscalar(x));
p.addOptional('nbins', 50, @(x) isnumeric(x) && isscalar(x));
p.parse(varargin{:});
par = p.Results;

% keep only selected clusters
K = size(Sigma, 3);
colors = hsv(K);
if ~ischar(par.clusters)
    keep = ismember(assignment, par.clusters);
    b(~keep, :) = [];
    assignment(~keep) = [];
end
[par.clusters, ~, assignment] = unique(assignment);
K = numel(par.clusters);
colors = colors(par.clusters, :);

figure(par.figure), clf
bg = 0.5 * ones(1, 3);
set(gcf, 'color', bg)
for i = 1 : K
    for j = [1 : i - 1, i + 1 : K]
        xi = b(assignment == i, :);
        xj = b(assignment == j, :);
        
        % model parameters for current pair of clusters
        ii = par.clusters(i);
        jj = par.clusters(j);
        mi = mu(ii, :);
        mj = mu(jj, :);
        Ci = Sigma(:, :, ii);
        Cj = Sigma(:, :, jj);
        S = (priors(ii) * Ci + priors(jj) * Cj) / (priors(ii) + priors(jj));
        
        % project on optimal axis
        w = S \ (mj - mi)';
        qi = xi * w - mj * w;
        qj = xj * w - mj * w;
        sd = std(qj);
        qi = qi / sd;
        qj = qj / sd;
        if mj * w > mi * w
            qi = -qi;
            qj = -qj;
        end
        
        % plot histograms on optimal axis
        axes('Position', [j - 1, K - i, 1, 1] / K) %#ok<LAXES>
        bins = linspace(-3, 10, par.nbins);
        h = [hist(qj, bins); hist(qi, bins)];
        hdl = bar(bins(2:end-1), h(:,2:end-1)', 1, 'stacked', 'linestyle', 'none');
        set(hdl(1), 'facecolor', colors(j, :))
        set(hdl(2), 'facecolor', colors(i, :))
        axis tight off
        ylim([0, 1.2 * max(h(1, :))])
    end
end

% plot grid on top
axes('position', [0 0 1 1])
hold on
plot([0 1], (1./ [K; K]) * (1 : K), 'k')
plot((1./ [K; K]) * (1 : K), [0 1], 'k')
axis([0 1 0 1])
axis off
