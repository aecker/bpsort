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
T = 1;  % minutes

% file = '/kyb/agmbrecordings/raw/Dennis/2014-08-01_12-35-36/2014-08-01_12-35-43/Electrophysiology%d.h5';
% file = '/kyb/agmbrecordings/raw/Charles/2014-06-26_13-17-59/2014-06-26_13-20-39/Electrophysiology%d.h5';
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
br = baseReader(file, 's1c*');
Fs = getSamplingRate(br);
fr = filteredReader(br, filterFactory.createHighpass(400, 600, Fs));
V = fr(1 : Fs * T * 60, :);
V = toMuV(br, V);


%% detect and sort spikes in groups
df = 5;
results = struct('s', {}, 'w', {}, 'b', {}, 'model', {});
for i = 1 : nGroups
    xi = V(:, groups(i, :));
    [s, t] = detectSpikes(xi, Fs);
    w = extractWaveforms(xi, s, -8 : 19);
    b = extractFeatures(w);
    model = MixtureModel.fit(b, df);
    
    results(i).s = s;
    results(i).w = w;
    results(i).b = b;
    results(i).model = model;
end


%% remove doubles
X = removeDuplicateClusters(results, 0.5, round(Fs * T * 60));


%% extract spikes
self = BP('window', [-.4 1.2], 'Fs', Fs);
q = round(self.tempFiltLen / 1000 * self.Fs);
W = BP.estimateWaveforms(V, X, self.samples);
R = BP.residuals(V, X, W, self.samples);
Vw = BP.whitenData(V, R, q);
Ww = BP.estimateWaveforms(Vw, X, self.samples, self.pruning);
Xn = BP.estimateSpikes(Vw, X, Ww, self.samples, self.upsampling);
Wn = BP.estimateWaveforms(V, Xn, self.samples);

