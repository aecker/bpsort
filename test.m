% Test harness using synthetic data

rng(1)
T = 2;          % min
K = 8;          % channels
M = 6;          % single units
Fs = 12000;     % Hz
N = T * 60 * Fs;
Kw = 3;         % size of spatial filter
rms = 6.5;      % RMS noise in 600-6000 Hz band
refrac = 2;     % ms refractory period 
spike = [0 10 18 10 -25 -60 -35 -11 0 7 10 12 13 13 12 10 7 3 1 0]';
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
    s = find(rand(N, 1) < rate(i) / Fs);
    prev = 0;
    for j = 1 : numel(s)
        if ~prev || s(j) - s(prev) > refrac / 1000 * Fs;
            v(s(j) + ndx, :) = v(s(j) + ndx, :) + spike * ampl(i, :);
        end
    end
    spikes{i} = s;
end


