function [t, w] = detectSpikes(V, Fs, threshold, win)
% Detect spikes and extract waveforms.
%   [t, w] = detectSpikes(V, Fs, threshold, win) detects spikes in V by
%   using threshold crossing with the supplied threshold (n times SD of the
%   noise). The sampling rate of the signal is Fs. Waveforms are extracted
%   using the specified window (in samples). The outputs are the spike
%   times in ms (t, a column vector) and waveforms (w, a 3d array of size
%   length(win) x #spikes x #channels). By convention the time of the
%   zeroth sample is 0 ms.

% detect local minima where at least one channel is above threshold
noiseSD = median(abs(V)) / 0.6745;
z = bsxfun(@rdivide, V, noiseSD);
mz = min(z, [], 2);
r = sqrt(sum(V .^ 2, 2));
dr = diff(r);
s = find(mz(2 : end - 1) < -threshold & dr(1 : end - 1) > 0 & dr(2 : end) < 0) + 1;

% remove spikes close to boundaries
s = s(s > win(1) & s < size(V, 1) - win(end));

% if multiple spikes occur within 1 ms we keep only the largest
refractory = 1 / 1000 * Fs;
N = numel(s);
keep = true(N, 1);
last = 1;
for i = 2 : N
    if s(i) - s(last) < refractory
        if r(s(i)) > r(s(last))
            keep(last) = false;
            last = i;
        else
            keep(i) = false;
        end
    else
        last = i;
    end
end
s = s(keep);
t = s / Fs * 1000; % convert to real times in ms

% extract waveforms
idx = bsxfun(@plus, s, win)';
w = reshape(V(idx, :), [length(win) numel(s) size(V, 2)]);
