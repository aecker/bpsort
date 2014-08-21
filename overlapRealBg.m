% Inject two highly overlapping clusters into a background of real data


%% Read probe configuration
config = 'V1x32-Poly2';
fid = fopen(config);
if fid
    channels = fscanf(fid, '%d');
    fclose(fid);
else
    error('Could not open config file %s!', config)
end


%% Load raw data
T = 5;  % minutes
Tstart = 3;
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
% more strongly correlated channels, but overall smaller spikes
% cstr = arrayfun(@(c) sprintf('s1c%d', c), channels(1 : 4), 'uni', false);
% a bit less correlated, but larger spikes
cstr = arrayfun(@(c) sprintf('s1c%d', c), channels(22 : 25), 'uni', false);
br = baseReader(file, cstr);
Fs = round(getSamplingRate(br));
fr = filteredReader(br, filterFactory.createHighpass(400, 600, Fs));
V = fr(Tstart * Fs * 60 + (1 : T * Fs * 60), :);
V = toMuV(br, V);

% downsample
Nyq = 6000;
[p, q] = rat(2 * Nyq / Fs);
V = resample(V, p, q);
Fs = p / q * Fs;
N = size(V, 1);
V_ = V;


%% Initialize using Gaussian mixture model
df = 5;
rng(1)
[s, t] = detectSpikes(V, Fs);
w = extractWaveforms(V, s);
b = extractFeatures(w);
model = MixtureModel.fit(b, df);
cl = model.cluster(b);
X0 = sparse(s, cl, 1, N, model.K);
X0_ = X0;


%% fit BP model
pass = [600 5000] / (Fs / 2);   % passband
bp = BP('window', [-1 2], 'Fs', Fs, 'passband', pass, 'tempFiltLen', 0.5);
[X, W] = bp.fit(V, X0, 1);
W_ = W;


%% Create a couple of artificial neurons
V = V_;
W = W_;
X0 = X0_;
rng(2)
[D, K, M] = size(W);
[~, ndx] = min(min(W(:, 3, :), [], 1));

% (a) slightly distort an existing waveform to create a new cluster that
%     overlaps with an existing, real neuron
W = W(:, :, [setdiff(1 : M, ndx), ndx]);
X0 = X0(:, [setdiff(1 : M, ndx), ndx]);
win = gausswin(11, 1);
win = win / sum(win);
distortion = exp(0.8 * conv2(randn(D, K), win, 'same'));
W(:, :, M + 1) = distortion .* W(:, :, M);

% (b) swap channels and distort waveforms to create two artificial clusters
%     that overlap, but don't overlap with any real neuron
swap = [3 4 1 2];
W(:, :, M + 2) = W(:, swap, M);
distortion = exp(conv2(randn(D, K), win, 'same'));
W(:, :, M + 3) = distortion .* W(:, swap, M);

% Inject spikes into real trace
p = 20;
Wu = reshape(resample(W, p, 1), [D * p, K, M + 3]);
rate = 3;
smp = bp.samples;
refrac = 2;
for i = 1 : 3
    s = find(rand(N - D, 1) < rate / Fs) - smp(1);
    jit = fix(rand(size(s)) * p);
    s = s + jit / p;
    viol = diff(s) < refrac / 1000 * Fs;
    while any(viol)
        s(viol) = [];
        viol = diff(s) < refrac / 1000 * Fs;
    end
    for j = 1 : numel(s)
        V(fix(s(j)) + smp, :) = V(fix(s(j)) + smp, :) + Wu(p - jit(j) : p : end, :, M + i);
    end
    X0(round(s), M + i) = 1; %#ok
end


%% plot waveforms
sp = max(abs(W(:)));
c = [1 0 0; 0 0.4 1; 1 0.5 0.5; 0 0.6 0.3];
figure(1), clf
h(1) = subplot(121); hold on
for i = 1 : 2
    plot(smp, bsxfun(@plus, W(:, :, end - 4 + i), (1 : K) * sp), 'color', c(i, :))
end
h(2) = subplot(122); hold on
for i = 1 : 2
    plot(smp, bsxfun(@plus, W(:, :, end - 2 + i), (1 : K) * sp), 'color', c(2 + i, :))
end
linkaxes(h)
axis([smp(1), smp(end), sp * [0, 4.5]])


%% Re-fit BP model
[X, W] = bp.fit(V, X0, 1);


%% plot raw trace with detected and assigned spikes
figure(2), clf
ndx = 1 : 1e5;
spacing = 200;
plot(bsxfun(@plus, V(ndx, :), (1 : K) * spacing), 'k')
hold on
p = bp.upsamplingFactor;
Wu = reshape(resample([zeros(1, K, M + 3); W], p, 1), [(D + 1) * p, K, M + 3]);
for i = 1 : 4
    spikes = find(X0(ndx, end - 4 + i));
    plot(spikes, ones(size(spikes)) * (ceil(i / 2) * 2 - 1.5) * spacing, '*', 'color', c(i, :), 'markersize', 10)
    [spikes, ~, x] = find(X(ndx, end - 4 + i));
    plot(spikes + x - 1, ones(size(spikes)) * (ceil(i / 2) * 2 - 1.5) * spacing, 'o', 'color', c(i, :), 'markersize', 10)
    ii = i + sign(mod(i, 2) - 0.5);
    for j = 1 : numel(spikes)
        shift = round((x(j) - 1) * p);
        plot(spikes(j) + smp, bsxfun(@plus, Wu(p - shift + (1 : p : D * p), :, M - 1 + ii), (1 : K) * spacing), 'color', c(ii, :))
        plot(spikes(j) + smp, bsxfun(@plus, Wu(p - shift + (1 : p : D * p), :, M - 1 + i), (1 : K) * spacing), 'color', c(i, :))
    end
end


%% Re-fit mixture model
rng(1)
[s, t] = detectSpikes(V, Fs);
w = extractWaveforms(V, s);
b = extractFeatures(w);
model = MixtureModel.fit(b, 2);
cl = model.cluster(b);

plotWaveformsTMP(w, cl)
plotSeparationTMP(b, model.mu, model.Sigma, model.pi, cl)





%% plot overlap using ground truth
dw = diff(reshape(W, D * K, 2), [], 2);
dw = dw / norm(dw);
ww = reshape(permute(w, [1 3 2]), D * K, []);
figure(3), clf
hist(dw' * ww, 80)
