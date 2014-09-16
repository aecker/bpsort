function [W, order] = orderTemplates(W, layout)
% Order waveform templates spatially by peak location.

M = size(W, 3);
y(layout.channels) = layout.y;
[~, order] = sort(y, 'descend');
mag = zeros(1, M);
peak = zeros(1, M);
for m = 1 : M
    Wi = mean(W(:, order, m, :), 4);
    [mag(m), peak(m)] = max(max(Wi) - min(Wi));
end
[~, order] = sort(peak * 1e6 - mag);
W = W(:, :, order, :);
