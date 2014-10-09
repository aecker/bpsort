function b = extractFeatures(w, q)
% Extract features for spike sorting.
%   b = extractFeatures(w, q) extracts the first q principal components
%   from the waveforms in w, which is a 3d array of size #samples x #spikes
%   x #channels. The output b is a matrix of size #spikes x #features.
%
%   We do PCA on the waveforms of each channel separately and keep q
%   principal components per channel.

[~, n, k] = size(w);
w = bsxfun(@minus, w, mean(w, 2));  % center data
b = zeros(n, k * q);
for i = 1:k
    C = w(:, :, i) * w(:, :, i)';   % covariance matrix
    [V, ~] = eigs(C, q);            % first q eigenvectors
    b(:, (1 : q) + q * (i - 1)) = w(:, :, i)' * V;
end
