function [mu, sigma, prior, assignment] = MoG1d(x, K, iter)
% One-dimensional mixture of two Gaussians

mu = randn(K, 1) * std(x) + mean(x);
sigma = ones(K, 1) * std(x);
prior = ones(K, 1) / K;

for i = 1 : iter
    % E step
    like = zeros(size(x, 1), K);
    for k = 1 : K
        like(:, k) = prior(k) * normal(x, mu(k), sigma(k));
    end
    p = sum(like, 2);
    post = bsxfun(@rdivide, like, p);
    post(p == 0, :) = 0;
    
    % M step
    Nk = sum(post, 1);
    for k = 1 : K
        mu(k) = x' * post(:, k) / Nk(k);
        xmu = x - mu(k);
        sigma(k) = xmu' * (xmu .* post(:, k)) / Nk(k);
    end
    prior = Nk / sum(Nk);
end
[~, assignment] = max(post, [], 2);


function p = normal(x, mu, sigma)
% Multivariate normal probability density
%   p = normal(x, mu, sigma) calculates the density of the normal
%   distribution with mean mu and variance sigma at x.

p = exp(-0.5 * (x - mu) .^ 2 / sigma) / sqrt(2 * pi * sigma);
