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
bp = BP('dt', 10, 'driftRate', 0.1, 'sigmaAmpl', 0.1);
[X, W] = bp.fit(V, X0, 1);


%% plot raw trace with detected and assigned spikes
[~, K, M, Nt] = size(W);
Tt = bp.dt * bp.Fs;
figure(2), clf
ndx = 1 : 1e6;
spacing = 200;
smp = bp.samples;
plot(bsxfun(@plus, V(ndx, :), (1 : K) * spacing), 'k')
hold on
c = hsv(M + 1);
for i = 1 : M
    Xn = X(ndx, i);
    spikes = find(Xn);
    ampl = Xn(spikes);
    for j = 1 : numel(spikes)
        t = ceil(spikes(j) / Tt);
        plot(spikes(j) + smp, bsxfun(@plus, ampl(j) * W(:, :, i, t), (1 : K) * spacing), 'color', c(i, :))
    end
end
set(gca, 'color', 0.5 * ones(1, 3))
colormap(c)
set(colorbar, 'ytick', (1 : M) + 0.5, 'yticklabel', 1 : M)


