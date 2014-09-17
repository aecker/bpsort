% Semi-manual model tweaking


%% Read probe configuration
configFile = 'V1x32-Poly2';
config = readConfig(configFile);


%% load data
T = 2;  % minutes

% file = '/kyb/agmbrecordings/raw/Dennis/2014-08-01_12-35-36/2014-08-01_12-35-43/Electrophysiology%d.h5';
% file = '/kyb/agmbrecordings/raw/Charles/2014-06-26_13-17-59/2014-06-26_13-20-39/Electrophysiology%d.h5';
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
br = baseReader(file, 's1c*');
Fs = getSamplingRate(br);
fr = filteredReader(br, filterFactory.createHighpass(400, 600, Fs));
V = fr(1 : Fs * T * 60, :);
V = toMuV(br, V);

% downsample
Nyq = 6000;
[p, q] = rat(2 * Nyq / Fs);
V = resample(V, p, q);
Fs = p / q * Fs;


%% Initialization
init


%% Model refinement

% the following are saved data files since the above init was run on a
% different computer
load modelfit
W_ = W;
load waveforms_whitened
Ww = W;
W = W_;  % ordered

% restore original channel ordering
% ONLY NECESSARY BECAUSE I RAN init WITH THE CHANNELS ORDERED, I.E.
%   [X, W] = bp.fit(V(:, config.channels, X0)
Ww(:, config.channels, :, :) = Ww;
W(:, config.channels, :, :) = W;

% prune waveforms
Ww = pruneWaveforms(Ww, config, 55, 2);

% remove templates that got pruned out completely
Wz = squeeze(sum(abs(Ww(:, :, :, 1)), 1) > 0);
ndx = any(Wz);
Ww = Ww(:, :, ndx, :);
W = W(:, :, ndx, :);
X = X(:, ndx);

% order waveform templates by peak location and amplitude
[Ww, order] = orderTemplates(Ww, config);
X = X(:, order);
W = W(:, :, order, :);


%% re-estimate spikes based on new set of templates
bp = BP;
R = bp.residuals(V, X, W);
Vw = bp.whitenData(V, R);
X = bp.estimateSpikes(Vw, X, Ww);


%% move artifacts and clear multi unit to the end

% load results from step above which was run on different computer
load pruned_results

M = size(X, 2);

% determined by manual inspection of waveforns
artifacts = [20 31 33 35 39 61:67];

order = [setdiff(1 : M, artifacts) artifacts];
X = X(:, order);
Xn = Xn(:, order);
W = W(:, :, order, :);
Ww = Ww(:, :, order, :);

% determined by manual inspection of waveforns and CCGs
sua = {1 2 4 5 7 9 10 [11 12] 13 14 15 16 17 19 21 22 [23 24] 27 32 (39 : 44) 45 48 [52 55] 56 60 61};

order = [sua{:}, setdiff(1 : M, [sua{:}])];
X = X(:, order);
Xn = Xn(:, order);
W = W(:, :, order, :);
Ww = Ww(:, :, order, :);

sua = [0 cumsum(cellfun(@numel, sua))];


%% single unit spike times for CCGs
Nsua = numel(sua) - 1;
t = cell(1, Nsua);
cl = cell(1, Nsua);
for m = 1 : Nsua
    [i, ~] = find(X(:, sua(m) + 1 : sua(m) + 1));
    t{m} = i / Fs * 1000;
    cl{m} = repmat(m, numel(i), 1);
end
[t, order] = sort(cat(1, t{:}));
cl = cat(1, cl{:});
cl = cl(order);


%% apply pruning pattern to non-whitened waveforms
Wz = squeeze(sum(abs(Ww(:, :, :, 1)), 1) > 0);
Wp = W;
for m = 1 : M
    Wp(:, ~Wz(:, m), m, :) = 0;
end


%% Plot waveforms
plotWaveforms(Wp(:, (config.channels), sua(1 : end - 1) + 1, end), 'spacing', 50, 'minsize', 0)


%% Plot traces with spikes overlaid
s = 1219200;
N = 20000;
plotTracesWithSingleUnits(V, X, Wp, bp, config, sua, s, N, 12)

