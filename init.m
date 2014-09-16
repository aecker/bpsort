% Initialization using traditional spike detection + clustering approach


%% Read probe configuration
configFile = 'V1x32-Poly2';
config = readConfig(configFile);


%% Create channel groups
num = 5;
overlap = 4;
ndx = bsxfun(@plus, 1 : num, (0 : (num - overlap) : numel(channels) - num)');
groups = channels(ndx);
nGroups = size(groups, 1);


%% Load raw data
T = 10;  % minutes

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


%% detect and sort spikes in groups
df = 5;
results = struct('s', {}, 'w', {}, 'b', {}, 'model', {});
for i = 1 : nGroups
    xi = V(:, groups(i, :));
    [s, t] = detectSpikes(xi, Fs);
    w = extractWaveforms(xi, s);
    b = extractFeatures(w);
    model = MixtureModel.fit(b, df);
    
    results(i).s = s;
    results(i).w = w;
    results(i).b = b;
    results(i).model = model;
end


%% remove doubles
X = keepMaxClusters(results, round(Fs * T * 60), 0.6);


%% extract spikes
pass = [600 5000] / (Fs / 2);   % passband
bp = BP('window', [-1 2], 'Fs', Fs, 'passband', pass, 'tempFiltLen', 0.5);
[X, W] = bp.fit(V, X, 3);
