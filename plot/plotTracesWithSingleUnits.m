function plotTracesWithSingleUnits(V, X, W, bp, layout, sua, start, N, spacing)
% Plot raw traces with single unit spikes overlaid.

if nargin < 9
    spacing = 15;
end
y(layout.channels) = layout.y;
[~, order] = sort(y);
Nsua = numel(sua) - 1;
K = size(V, 2);
Tt = bp.dt * bp.Fs;
ndx = start + (1 : N);
V = V(ndx, order);
W = W(:, order, :, :);
R = bp.residuals(V, X(ndx, :), W);
sd = std(R);
V = bsxfun(@rdivide, V, sd);
W = bsxfun(@rdivide, W, sd);

figure(2), clf
smp = bp.samples;
plot(bsxfun(@plus, V, (1 : K) * spacing), 'k')
hold on
Nc = 11;
k = 1;
c = hsv(Nc);
cc = zeros(Nsua, 3);
for i = 1 : Nsua
    for m = sua(i) + 1 : sua(i + 1);
        channels = find(sum(abs(W(:, :, m, 1)), 1));
        Xm = X(ndx, m);
        spikes = find(Xm);
        ampl = real(Xm(spikes));
        for j = 1 : numel(spikes)
            t = ceil(spikes(j) / Tt);
            plot(spikes(j) + smp, bsxfun(@plus, ampl(j) * W(:, channels, m, t), channels * spacing), 'color', c(k, :))
        end
    end
    cc(i, :) = c(k, :);
    k = rem(k + 2, Nc) + 1;
end
set(gca, 'color', 0.6 * ones(1, 3), 'xlim', [0 N], 'tickdir', 'out', ...
    'ylim', [0, K + 1] * spacing, 'ytick', (2 : 2 : K) * spacing, ...
    'yticklabel', 2 : 2 : K, 'ticklen', [0.005 0], 'box', 'off')
colormap(cc)
hc = colorbar;
set(hc, 'ytick', (1 : Nsua) + 0.5, 'yticklabel', 1 : Nsua, 'ydir', 'reverse', 'ticklen', [0 0])
ylabel('Channel')
xlabel('Time (ms)')
