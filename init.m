% Initialization using traditional spike detection + clustering approach


%% Read probe configuration
config = 'V1x32-Poly2';
fid = fopen(config);
if fid
    channels = fscanf(fid, '%d');
    fclose(fid);
else
    error('Could not open config file %s!', config)
end


%% Create channel groups
num = 4;
overlap = 2;
ndx = bsxfun(@plus, 1 : num, (0 : overlap : numel(channels) - num)');
groups = channels(ndx);
nGroups = size(groups, 1);


%% Load raw data
T = 5;  % minutes

% file = '/kyb/agmbrecordings/raw/Dennis/2014-08-01_12-35-36/2014-08-01_12-35-43/Electrophysiology%d.h5';
% file = '/kyb/agmbrecordings/raw/Charles/2014-06-26_13-17-59/2014-06-26_13-20-39/Electrophysiology%d.h5';
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
br = baseReader(file, 's1c*');
Fs = getSamplingRate(br);
Nyq = 6000;
fr = filteredReader(br, filterFactory.createBandpass(400, 600, 5800, Nyq, Fs));
x = fr(1 : Fs * T * 60, :);
x = toMuV(br, x);

% downsample
[p, q] = rat(2 * Nyq / Fs);
x = resample(x, p, q);
Fs = p / q * Fs;


%% detect and sort spikes in groups
df = 2;  % use mixture of t model with df = 2
results = struct('s', {}, 'w', {}, 'b', {}, 'model', {});
for i = 1 : nGroups
    xi = x(:, groups(i, :));
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



