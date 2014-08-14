function X = removeDuplicateClusters(results, threshold, T)
% Remove duplicate clusters

M = sum(arrayfun(@(r) numel(r.model.pi), results));
R = numel(results);
spikes = cell(1, R);
clusters = cell(1, R);
m = zeros(1, M);
k = 0;
for i = 1 : R
    r = results(i);
    a = cluster(r.model, r.b);
    J = numel(r.model.pi);
    for j = 1 : J
        m(k + j) = mean(mean(mean(r.w(:, a == j, :), 2) .^ 2));
    end
    spikes{i} = r.s;
    clusters{i} = k + a;
    k = k + J;
end
spikes = cat(1, spikes{:});
clusters = cat(1, clusters{:});

% order cluster ids by magnitude of average waveform
[~, order] = sort(m, 'descend');
for i = 1 : M
    clusters(clusters == order(i)) = -i;
end
clusters = -clusters;
total = hist(clusters, 1 : M);

% order spikes in time
[spikes, order] = sort(spikes);
clusters = clusters(order);

% remove spikes of smaller size
N = numel(spikes);
keep = true(N, 1);
prev = 1;
refrac = 4; % 1/3 ms
for i = 2 : N
    if spikes(i) - spikes(prev) < refrac
        if clusters(i) > clusters(prev)
            keep(i) = false;
        else
            keep(prev) = false;
            prev = i;
        end
    else
        prev = i;
    end
end
spikes = spikes(keep);
clusters = clusters(keep);

% remove clusters that lost too many spikes to other clusters
frac = hist(clusters, 1 : M) ./ total;
keep = true(numel(spikes), 1);
for i = 1 : M
    if frac(i) < threshold
        keep(clusters == i) = false;
    end
end
spikes = spikes(keep);
clusters = clusters(keep);
[~, ~, clusters] = unique(clusters);

% create spike matrix
X = sparse(spikes, clusters, 1, T, max(clusters));
