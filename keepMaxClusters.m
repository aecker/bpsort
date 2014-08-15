function X = keepMaxClusters(results, T)
% Keep those clusters that have largest waveform on the center channel

K = size(results(1).w, 3);
center = (K + 1) / 2;

R = numel(results);
spikes = {};
for i = 1 : R
    r = results(i);
    a = cluster(r.model, r.b);
    for j = 1 : numel(r.model.pi);
        m = permute(sum(mean(r.w(:, a == j, :), 2) .^ 2, 1), [3 2 1]);
        [~, ndx] = max(m);
        if ndx == center || (i == 1 && ndx < center) || (i == R && ndx > center)
            spikes{end + 1} = r.s(a == j); %#ok
        end
    end
end

M = numel(spikes);
X = sparse(T, M);
for i = 1 : M
    X(spikes{i}, i) = 1; %#ok
end
