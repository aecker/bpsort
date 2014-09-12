% Tracking of waveform drift using a Kalman filter


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
Tstart = 2;
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
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


%% Estimate waveforms
bp = BP('dt', 10, 'driftRate', 0.1, 'sigmaAmpl', 0.05);
[X, W] = bp.fit(V, X0, 1);


%% plot waveforms
M = size(W, 3);
c = hsv(M);
figure(1)
set(gcf, 'DefaultAxesColorOrder', c);
clf, hold on
set(gca, 'color', 0.5 * ones(1, 3));
plot(bsxfun(@plus, (1 : M) * 100, reshape(W(:, :, :, 1), [], M)))
colormap(c)
set(colorbar, 'ytick', (1 : M) + 0.5, 'yticklabel', 1 : M)


%% plot raw trace with detected and assigned spikes
[~, K, M, Nt] = size(W);
Tt = bp.dt * bp.Fs;
figure(2), clf
ndx = 1 : 1e6;
spacing = 200;
smp = bp.samples;
plot(bsxfun(@plus, V(ndx, :), (1 : K) * spacing), 'k')
hold on
c = hsv(M);
for i = 1 : M
    Xn = X(ndx, i);
    spikes = find(Xn);
    ampl = real(Xn(spikes));
    for j = 1 : numel(spikes)
        t = ceil(spikes(j) / Tt);
        plot(spikes(j) + smp, bsxfun(@plus, ampl(j) * W(:, :, i, t), (1 : K) * spacing), 'color', c(i, :))
    end
end
set(gca, 'color', 0.5 * ones(1, 3))
colormap(c)
set(colorbar, 'ytick', (1 : M) + 0.5, 'yticklabel', 1 : M)


%% Test splitting based on bimodal amplitude distributions
% X_ = X;
M = size(X, 2);
K = 2;
iter = 100;
mu = zeros(M, K);
sigma = zeros(M, 1);
prior = zeros(M, K);
cl = cell(M, 1);
bic = zeros(M, 2);
for i = 1 : M
    a = full(real(X(X(:, i) > 0, i)));
    [mu(i, :), sigma(i), prior(i, :), cl{i}, bic(i, 2)] = mog1d(a, K, iter);
    
    % BIC for single Gaussian
    sd = var(a);
    bic(i, 1) = sum((a - mean(a)) .^ 2 / sd + log(2 * pi * sd)) + 2 * log(numel(a));
end
dprime = abs(diff(mu, [], 2)) ./ sqrt(sigma);


%%
split = find(bic(:, 1) > bic(:, 2) & dprime > 1);
for i = split'
    ndx = find(X(:, i));
    X(ndx(cl{i} == 2), end + 1) = X(ndx(cl{i} == 2), i); %#ok
    X(ndx(cl{i} == 2), i) = 0;
end


%% re-fit
[X, W] = bp.fit(V, X, 1);

