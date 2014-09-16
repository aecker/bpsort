function plotWaveforms(W, varargin)
% Plot waveforms.

p = inputParser;
p.KeepUnmatched = false;
p.addOptional('minsize', 10);  % muV peak-to-peak amplitude
p.addOptional('spacing', 50);  % muV vertical spacing
p.addOptional('upsampling', 5); % upsampling factor for plotting
p.parse(varargin{:});
args = p.Results;

[D, K, M] = size(W);
sx = 2 * D;
Nc = 10;

figure
hold on
axis([sx * [-0.07, M], 0, (K + 1) * args.spacing]);
plot(ones(2, 1) * (0.93 : M - 1) * sx, ylim, 'color', 0.6 * ones(1, 3))
c = hsv(Nc);
for i = 1 : M
    plot((i - 1) * sx + [0.07 0.5] * sx, ones(2, 1) * (1 : 2 : K) * args.spacing, ':', 'color', 0.8 * ones(1, 3))
    plot((i - 1) * sx + [0.5 0.85] * sx, ones(2, 1) * (2 : 2 : K) * args.spacing, ':', 'color', 0.8 * ones(1, 3))
    p2p = max(W(:, :, i)) - min(W(:, :, i));
    ndx = find(p2p > args.minsize);
    ndxL = ndx(mod(ndx, 2) == 1);
    if ~isempty(ndxL)
        plot((1 : 1 / args.upsampling : D + 0.99) + sx * (i - 1), ...
             bsxfun(@plus, resample(W(:, ndxL, i), args.upsampling, 1), ndxL * args.spacing), ...
             'color', c(rem(i - 1, Nc) + 1, :))
    end
    ndxR = ndx(mod(ndx, 2) == 0);
    if ~isempty(ndxR)
        plot((1 : 1 / args.upsampling : D + 0.99) + sx * (i - 1) + 0.75 * D, ...
             bsxfun(@plus, resample(W(:, ndxR, i), args.upsampling, 1), ndxR * args.spacing), ...
             'color', c(rem(i - 1, Nc) + 1, :))
    end
end
set(gca, 'tickdir', 'out', 'xtick', sx * (0.5 : M), 'xticklabel', 1 : M, ...
    'ytick', (1 : 2 : K) * args.spacing, 'yticklabel', 1 : 2 : K)
xlabel('Cluster')
ylabel('Channel')
