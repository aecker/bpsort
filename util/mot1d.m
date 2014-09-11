function [mu, sigma, prior, assignment] = mot1d(x, df, K, iter)
% One-dimensional mixture of Student's t distributions.

mu = linspace(min(x), max(x), K);
sigma = ones(K, 1) * std(x);
prior = ones(K, 1) / K;

for i = 1 : iter
    
    % E step
    like = zeros(size(x, 1), K);
    for k = 1 : K
        like(:, k) = prior(k) * tdist(x, mu(k), sigma(k), df);
    end
    p = sum(like, 2);
    post = bsxfun(@rdivide, like, p);
    post(p == 0, :) = 0;
    
    % M step
    Nk = sum(post, 1);
    for k = 1 : K
        uk = (df + 1) ./ (df + (x - mu(k)) .^ 2 / sigma(k));
        mu(k) = x' * (post(:, k) .* uk) / Nk(k);
        xmu = x - mu(k);
        sigma(k) = xmu' * (xmu .* post(:, k) .* uk) / Nk(k);
    end
    prior = Nk / sum(Nk);
end
[~, assignment] = max(post, [], 2);


function p = tdist(x, mu, sigma, df)
% Student's t probability density
%   p = tdist(x, mu, sigma, df) calculates the density of the t
%   distribution with scale parameter sigma and df degrees of freedom at x.

delta = (x - mu) .^ 2 / sigma;
p = exp(gammaln((df + 1) / 2) - gammaln(df / 2) - log(df * pi * sigma) / 2 ...
        - ((df + 1) / 2) * log(1 + delta / df));
