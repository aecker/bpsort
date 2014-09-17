% Raw traces plot

T = 0.3;  % minutes

% file = '/kyb/agmbrecordings/raw/Dennis/2014-08-01_12-35-36/2014-08-01_12-35-43/Electrophysiology%d.h5';
% file = '/kyb/agmbrecordings/raw/Charles/2014-06-26_13-17-59/2014-06-26_13-20-39/Electrophysiology%d.h5';
file = '/kyb/agmbrecordings/raw/Charles/2014-07-21_13-50-16/2014-07-21_13-52-57/Electrophysiology%d.h5';
br = baseReader(file, 's1c*');
Fs = round(getSamplingRate(br));
fr = filteredReader(br, filterFactory.createHighpass(400, 600, Fs));

figure(1), clf

spacing = 200;

if ~exist('s', 'var') || ~isscalar(s) || s + T * 60 * Fs > length(fr)
    s = 0;
end

while s + T * 60 * Fs < length(fr)

    V = fr(s + (1 : Fs * T * 60), :);
    V = toMuV(br, V);
    
    % downsample
    Nyq = 6000;
    [p, q] = rat(2 * Nyq / Fs);
    V = resample(V, p, q);
    Fs = p / q * Fs;
    
    V = V(:, flipud(config.channels));

    K = size(V, 2);
    plot(bsxfun(@plus, V, (1 : K) * spacing), 'k')
    
    title(sprintf('s = %d', round(s)))
    axis([0, T * 60 * Fs, 0, (K + 1) * spacing])
    yt = [1, 4 : 4 : K];
    set(gca, 'ytick', yt * spacing, 'yticklabel', yt)
    
    pause
    
    s = s + T * 60 * Fs;
    
end
