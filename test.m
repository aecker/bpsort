% Test harness using synthetic data

rng(1)
T = 0.5;          % min
K = 8;          % channels
M = 6;          % single units
Fs = 12000;     % Hz
N = T * 60 * Fs;
Kw = 3;         % size of spatial filter
rms = 6.5;      % RMS noise in 600-6000 Hz band
refrac = 2;     % ms refractory period 
jitter = 10;    % jitter in spike timing (fraction of one sample)
spike = [0 10 18 10 -25 -60 -35 -11 0 7 10 12 13 13 12 10 7 3 1 0]';
spikeJit = [zeros(jitter / 2, 1); resample(spike, jitter, 1)];
D = numel(spike);

% generate 1/f noise
f = linspace(0, Fs, N)';
g = 1 ./ f;
g(1) = 0;
g(end - (0 : N / 2 - 2)) = g(2 : N / 2);
v = randn(N, K + 2 * Kw);
v = ifft(bsxfun(@times, fft(v), g));

% highpass filter
[b, a] = butter(5, 600 / Fs * 2, 'high');
v = filtfilt(b, a, v);

% spatial filter
a = 0.2;
Kw = 3;
w = a * gausswin(2 * Kw + 1)';
w(Kw + 1) = 1 - a;
v = conv2(v, w, 'valid');

% normalize
v = v / std(v(:)) * rms;

% add spikes
[~, peak] = min(spike);
ndx = (1 : numel(spike)) - peak;
ampl = diag(exp(randn(M, 1) * 0.7));
ampl(:, K) = 0;
w = gausswin(7, 3);
ampl = conv2(ampl, w', 'same');
rate = exp(randn(M, 1) + 2);
spikes = cell(1, M);
for i = 1 : M
    s = peak + find(rand(N - numel(spike), 1) < rate(i) / Fs);
    s = s + round(rand(size(s)) * jitter) / jitter;
    viol = diff(s) < refrac / 1000 * Fs;
    while any(viol)
        s(viol) = [];
        viol = diff(s) < refrac / 1000 * Fs;
    end
    for j = 1 : numel(s)
        start = round((1 - rem(s(j), 1)) * jitter);
        sj = fix(s(j));
        v(sj + ndx, :) = v(sj + ndx, :) + spikeJit(start + (1 : jitter : jitter * D)) * ampl(i, :);
    end
    spikes{i} = s;
end

X = sparse(N, M);
for i = 1 : numel(spikes)
    X(round(spikes{i}), i) = 1; %#ok
end
V = v;

% run initialized with ground truth
self = BP('window', [-.4 1.2]);
q = round(self.tempFiltLen / 1000 * self.Fs);
W = BP.estimateWaveforms(V, X, self.samples);
R = BP.residuals(V, X, W, self.samples);
Vw = BP.whitenData(V, R, q);
Ww = BP.estimateWaveforms(Vw, X, self.samples, self.pruning);
Xn = BP.estimateSpikes(Vw, X, Ww, self.samples, self.upsampling);

