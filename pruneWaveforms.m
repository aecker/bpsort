function W = pruneWaveforms(W, layout, radius, threshold)
% Prune waveforms, taking into account the spatial arrangement of channels.

ctrWeight = 0.7;
[~, K, M, ~] = size(W);

% smooth with adjacent channels
x(layout.channels) = layout.x;
y(layout.channels) = layout.y;
nrm = zeros(K, M);
N = 0;
for k = 1 : K
    neighbors = (x - x(k)) .^ 2 + (y - y(k)) .^ 2 < radius ^ 2;
    h = neighbors * (1 - ctrWeight) / (sum(neighbors) - 1);
    h(k) = ctrWeight;
    nrm(k, :) = sqrt(max(sum(sum(bsxfun(@times, h, W), 2) .^ 2, 1), [], 4));
    N = max(N, sum(neighbors) - 1);
end

% find contiguous region around maximum above threshold
for m = 1 : M
    [mx, peak] = max(nrm(:, m));
    active = false(1, K);
    if mx > threshold
        neighbors = peak;
        active(neighbors) = true;
    else
        neighbors = [];
    end
    while ~isempty(neighbors)
        newNeighbors = false;
        for k = neighbors
            newNeighbors = newNeighbors | ...
                (((x - x(k)) .^ 2 + (y - y(k)) .^ 2 < radius ^ 2) & nrm(:, m)' > threshold & ~active);
        end
        neighbors = find(newNeighbors);
        active(neighbors) = true;
    end
    
    % fill holes (channels below threshold where all neighbors are included)
    for k = 1 : K
        neighbors = (x - x(k)) .^ 2 + (y - y(k)) .^ 2 < radius ^ 2;
        neighbors(k) = false;
        active(k) = active(k) | sum(active(neighbors)) == N;
    end

    W(:, ~active, m, :) = 0;
end



