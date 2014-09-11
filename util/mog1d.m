function [mu, sigma, prior, assignment] = mog1d(x, K, iter)
% One-dimensional mixture of two Gaussians with common variance.

mu = linspace(min(x), max(x), K);
sigma = std(x);
prior = ones(K, 1) / K;

for i = 1 : iter
    % E step
    like = zeros(size(x, 1), K);
    for k = 1 : K
        like(:, k) = prior(k) * normal(x, mu(k), sigma);
    end
    p = sum(like, 2);
    post = bsxfun(@rdivide, like, p);
    post(p == 0, :) = 0;
    
    % M step
    Nk = sum(post, 1);
    dev = 0;
    for k = 1 : K
        mu(k) = x' * post(:, k) / Nk(k);
        xmu = x - mu(k);
        dev = dev + xmu' * (xmu .* post(:, k));
    end
    sigma = dev / numel(x);
    prior = Nk / sum(Nk);
end
[~, assignment] = max(post, [], 2);


function p = normal(x, mu, sigma)
% Multivariate normal probability density
%   p = normal(x, mu, sigma) calculates the density of the normal
%   distribution with mean mu and variance sigma at x.

p = exp(-0.5 * (x - mu) .^ 2 / sigma) / sqrt(2 * pi * sigma);
