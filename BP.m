% Spike sorting using binary pursuit.
%
% Implements the algorithm described in the following paper: 
% Pillow, Shlens, Chichilnisky & Simoncelli (2013): A Model-Based Spike
% Sorting Algorithm for Removing Correlation Artifacts in Multi-Neuron
% Recordings. PLoS ONE 8, e62123.
%
% AE 2014-08-07

classdef BP
    properties %#ok<*PROP>
        layout      % geometrical layout of the electrode (see Layout class)
        window      % window for extracting waveform: [a b] ms
        samples     % samples corresponding to waveform window
        Fs          % sampling rate
        verbose     % verbose output during fitting?
        tempFiltLen % length of temporal whitening filter (ms)
        upsamplingFactor  % upsampling factor for spike times
        upsamplingFilter        % filter used for subsampling
        upsamplingFilterOrder   % filter order (for subsampling filter)
        passband    % passband of continuous input signal
        dt          % time window for tracking waveform drift (sec)
        driftRate   % waveform drift rate (muV, SD per time step)
        sigmaAmpl   % SD of waveform amplitudes (muV)
        splitMinDPrime  % Min d' on aomplitudes for splitting a cluster
        splitMinPrior   % Min prior prob ob second component for splitting
        splitMinRate    % Min firing rate for splitting
        pruningRadius       % radius for smoothing before pruning
        pruningCtrWeight    % center weight of smoothing filter
        pruningThreshold    % pruning threshold
        D           % # dimensions
    end
    
    methods
        
        function self = BP(layout, varargin)
            % BP constructor
            %   bp = BP('param1', value1, 'param2', value2, ...) constructs
            %   a BP object with the following optional parameters:
            %
            %   window: 1x2 vector specifying the time window (ms) to
            %       extract waveforms (peak = 0; default [-0.5 1])
            %   Fs: sampling rate (Hz)
            %   verbose: true|false
            %   tempFiltLen: length of filter for temporal whitening
            %       (default = 0.7 ms)
            %   upsamplingFactor: upsampling factor used for spike
            %       detection (default = 5)
            %   pruning: constant applied for subset selection when
            %       estimating waveforms (default = 1)
            %   passband: passband of the continuous input signal (default:
            %       [600 15000] / Nyquist)
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('window', [-1 2]);
            p.addOptional('Fs', 12000);
            p.addOptional('verbose', false);
            p.addOptional('tempFiltLen', 0.5);
            p.addOptional('upsamplingFactor', 5, @(p) assert(mod(p, 2) == 1, 'Upsampling factor must be odd!'));
            p.addOptional('passband', [0.6 5] / 12);
            p.addOptional('dt', 20);
            p.addOptional('driftRate', 0.1);
            p.addOptional('sigmaAmpl', 0.05);
            p.addOptional('splitMinDPrime', 1);
            p.addOptional('splitMinPrior', 0.05);
            p.addOptional('splitMinRate', 0.1);
            p.addOptional('pruningRadius', 1);
            p.addOptional('pruningCtrWeight', 1);
            p.addOptional('pruningThreshold', 1);
            p.parse(varargin{:});
            self.window = p.Results.window;
            self.Fs = p.Results.Fs;
            self.samples = round(self.window(1) * self.Fs / 1000) : round(self.window(2) * self.Fs / 1000);
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
            self.tempFiltLen = p.Results.tempFiltLen;
            self.upsamplingFactor = p.Results.upsamplingFactor;
            self.passband = p.Results.passband;
            self.dt = p.Results.dt;
            self.driftRate = p.Results.driftRate;
            self.sigmaAmpl = p.Results.sigmaAmpl;
            self.splitMinDPrime = p.Results.splitMinDPrime;
            self.splitMinPrior = p.Results.splitMinPrior;
            self.splitMinRate = p.Results.splitMinRate;
            self.pruningRadius = p.Results.pruningRadius;
            self.pruningCtrWeight = p.Results.pruningCtrWeight;
            self.pruningThreshold = p.Results.pruningThreshold;
            
            % store or read electrode layout
            if isa(layout, 'Layout')
                self.layout = layout;
            else
                self.layout = Layout(layout);
            end
            
            % design filter for resampling
            p = self.upsamplingFactor;
            n = 5;
            len = 2 * n * p + 1;
            f = 1 / p;
            h = p * firls(len - 1, [0 f f 1], [1 1 0 0])' .* kaiser(len, 5);
            h = [zeros(fix(p / 2), 1); h; zeros(fix(p / 2), 1)];
            self.upsamplingFilter = reshape(h, p, 2 * n + 1)';
            self.upsamplingFilterOrder = n;
        end
        
        
        function [X, W] = fit(self, V, X, iter)
            % Fit model (i.e. estimate waveform templates and spike times).
            %   [X, W] = self.fit(V, X0) fits the model to waveforms V
            %   using the initial spike sorting results X0.
            %
            %   [X, W] = self.fit(V, X0, iter) uses the specified number of
            %   iterations to fit the parameters (default = 3).
            %
            %   INPUTS
            %
            %   V       Continuous voltage signal
            %           T-by-K      T: number of time bins
            %                       K: number of channels
            %
            %   X0      Initial spike sorting result (sparse matrix, where
            %           X_ij=1 indicates a spike at sample i and neuron j)
            %           T-by-M      M: number of clusters
            %
            %   iter    Number of iterations to run
            %
            %
            %   OUTPUTS
            %
            %   X       Spike times (same format as input X0)
            %
            %   W       Array of waveforms 
            %           D-by-K-by-M     D: number of samples
            %                           K: number of channels
            %                           M: number of neurons

            if nargin < 4
                iter = 2;
            end
            split = true;
            merged = true;
            i = 0;
            while i < iter && split && merged
                
                % estimate waveforms in whitened space
                W = self.estimateWaveforms(V, X);
                R = self.residuals(V, X, W);
                Vw = self.whitenData(V, R);
                Ww = self.estimateWaveforms(Vw, X);
                
                % merge templates that are too similar
                [X, Ww, merged] = self.mergeTemplates(X, Ww);

                % prune waveforms and estimate spikes
                Ww = self.pruneWaveforms(Ww);
                X = self.estimateSpikes(Vw, X, Ww);
                
                % split templates with bimodal amplitude distribution
                [X, split] = self.splitTemplates(X);
                
                % ensure we run a fixed number of iterations after the last
                % splitting and/or merging operation
                i = (~split & ~merged) * (i + 1);
            end
            
            % Re-estimate non-whitened waveforms and apply the same pruning
            % as to whitened waveforms
            W = self.estimateWaveforms(V, X);
            zero = max(sum(abs(Ww), 1), [], 4) < 1e-6;
            W = bsxfun(@times, W, zero);
        end
        
        
        function W = estimateWaveforms(self, V, X)
            % Estimate waveform templates given spike times.
            %   W = self.estimateWaveforms(V, X) estimates the waveforms W
            %   given the observed voltage V and the current estimate of
            %   the spike times X.
            
            [T, K] = size(V);
            M = size(X, 2);
            D = numel(self.samples);
            Tdt = self.dt * self.Fs;
            Ndt = ceil(T / Tdt);
            
            % Pre-compute convolution matrix: MX * W = conv(X, W)
            [i, j, x] = find(X);
            r = imag(x);
            a = real(x);
            d = 2 * (r > 0) - 1;
            i = [i; i + d];
            i = bsxfun(@plus, i, self.samples);
            valid = find(i > 0 & i <= T);
            j = bsxfun(@plus, (j - 1) * D, 1 : D);
            j = [j; j];
            x = repmat([a .* (1 - abs(r)); a .* abs(r)], 1, D);
            
            [i, order] = sort(i(valid));
            j = j(valid(order));
            x = x(valid(order));
            
            borders = zeros(1, Ndt + 1);
            for t = 1 : Ndt
                borders(t + 1) = find(i <= t * Tdt, 1, 'last');
            end
            
            W = zeros(D * M, K, Ndt);
            Q = eye(D * M) * self.driftRate ^ 2;
            
            % Pre-compute MX' * MX
            MXprod = zeros(D * M, D * M, Ndt);
            for t = 1 : Ndt
                idx = borders(t) + 1 : borders(t + 1);
                MXt = sparse(i(idx) - (t - 1) * Tdt, j(idx), x(idx), Tdt, D * M);
                MXprod(:, :, t) = MXt' * MXt;
            end
            
            % Initialize
            MX1 = sparse(i(1 : borders(2)), j(1 : borders(2)), x(1 : borders(2)), Tdt, D * M);
            n = full(sum(MX1, 1));
            idx = n > 0;
            W(idx, :, 1) = MXprod(idx, idx, 1) \ (MX1(:, idx)' * V(1 : Tdt, :));
            
            % Go through all channels
            for k = 1 : K
                
                % Initialize state covariance
                P = zeros(D * M, D * M, Ndt);
                P(:, :, 1) = diag(1 ./ (n + ~n));
            
                % Forward pass
                Pti = zeros(D * M, D * M, Ndt);
                I = eye(D * M);
                for t = 2 : Ndt
                    
                    % Predict
                    Pt = P(:, :, t - 1) + Q;
                    Pti(:, :, t) = inv(Pt);
                    Wt = W(:, k, t - 1);
                    
                    % Update
                    idx = borders(t) + 1 : borders(t + 1);
                    MXt = sparse(i(idx) - (t - 1) * Tdt, j(idx), x(idx), Tdt, D * M);
                    MXp = MXprod(:, :, t);
                    Kp = Pt * (I - MXp / (Pti(:, :, t) + MXp)); % Kalman gain (K = Kp * MX)
                    KpMXp = Kp * MXp;
                    tt = (t - 1) * Tdt + (1 : Tdt);
                    W(:, k, t) = Wt + Kp * (MXt' * V(tt, k)) - KpMXp * Wt;
                    P(:, :, t) = (I - KpMXp) * Pt;
                end
                
                % Backward pass
                for t = Ndt - 1 : -1 : 1
                    Ct = P(:, :, t) * Pti(:, :, t + 1);
                    W(:, k, t) = W(:, k, t) + Ct * (W(:, k, t + 1) - W(:, k, t));
                end
            end
            
            % Re-organize waveforms by cluster
            W = reshape(W, [D M K Ndt]);
            W = permute(W, [1 3 2 4]);
        end
        
        
        function V = whitenData(self, V, R)
            % Whiten data.
            %   V = self.whitenData(V, R) whitens the data V, assuming
            %   that the spatio-temporal covariance separates into a
            %   spatial and a temporal component. Whitening filters are
            %   estimated from the residuals R.

            % determine frequencies outside the passband to avoid
            % amplification of those frequencies
            q = round(self.tempFiltLen / 1000 * self.Fs);
            k = 4 * q + 1;
            F = linspace(0, 2, k + 1);
            F = F(1 : end - 1);
            high = find(F > self.passband(2) & F < 2 - self.passband(2));
            low = F < self.passband(1) | F > 2 - self.passband(1);
            U = dftmtx(k);
            
            % temporal whitening
            for i = 1 : size(V, 2)
                
                % construct filter for temporal whitening
                c = xcorr(R(:, i), 2 * q, 'coeff');
                c = ifftshift(c);
                ci = 1./ abs(fft(c));
                if ~isempty(high)
                    ci(high) = ci(high(1) - 1);
                end
                ci(low) = 0;
                w = real(U * (sqrt(ci) .* U(2 * q + 1, :)') / k);
                w = w(q + 1 : end - q);

                % apply temporal whitening filter
                V(:, i) = conv(V(:, i), w, 'same');
                R(:, i) = conv(R(:, i), w, 'same');
            end
            
            % spatial whitening
            V = V * chol(inv(cov(R)))';
        end
        
        
        function V = residuals(self, V, X, W)
            % Compute residuals by subtracting waveform templates.
            %   R = self.residuals(V, X, W) computes the residuals by
            %   subtracting the model prediction X * W from the data V.
            
            T = size(V, 1);
            Tdt = self.dt * self.Fs;
            for i = 1 : size(X, 2)
                spikes = find(X(:, i));
                for j = 1 : numel(spikes)
                    r = imag(X(spikes(j), i));
                    a = real(X(spikes(j), i));
                    s = sign(r);
                    t = ceil(spikes(j) / Tdt);
                    samples = spikes(j) + self.samples;
                    valid = samples > 0 & samples < T;
                    V(samples(valid), :) = V(samples(valid), :) - a * (1 - abs(r)) * W(valid, :, i, t);
                    samples = samples + s;
                    valid = samples > 0 & samples < T;
                    V(samples(valid), :) = V(samples(valid), :) - a * abs(r) * W(valid, :, i, t);
                end
            end
        end
        
        
        function Xn = estimateSpikes(self, V, X, W)
            % Estimate spike times given waveform templates.
            %   X = self.estimateSpikes(V, X, W) estimates the spike times
            %   given the current estimate of the waveforms using binary
            %   pursuit.

            [T, K] = size(V);
            M = size(X, 2);
            Tdt = self.dt * self.Fs;
            Ndt = ceil(T / Tdt);
            DL = zeros(T, M);
            A = zeros(T, M);
            wws = zeros(Ndt, M);
            wVs = zeros(T, M);
            p = self.upsamplingFactor;
            D = self.D;
            s = 1 - D : D - 1;
            dDL = zeros(2 * D - 1, M, M, p, Ndt);
            for t = 1 : Ndt
                Wt = W(:, :, :, t);
                
                % initialize \Delta L (Eq. 9) assuming X = 0 (no spikes)
                r = sum(X > 0, 1) / T;
                gamma = log(1 - r) - log(r);
                ww = permute(sum(sum(Wt .^ 2, 1), 2), [1 3 2]);
                convVW = 0;
                for k = 1 : K
                    Wk = permute(Wt(:, k, :), [1 3 2]);
                    Vk = V(max(1, (t - 1) * Tdt - self.samples(end) + 1) : min(T, t * Tdt - self.samples(1)), k);
                    convVWk = conv2(Vk, flipud(Wk));
                    first = (1 + (t > 1)) * self.samples(end) + 1;
                    last = size(convVWk, 1) + (1 + (t < Ndt)) * self.samples(1);
                    convVW = convVW + convVWk(first : last, :);
                end
                sa = 1 / self.sigmaAmpl ^ 2;
                At = bsxfun(@rdivide, convVW + sa, ww + sa);
                DLt = bsxfun(@minus, (At / 2) .* (convVW + sa), gamma + sa / 2);
                idx = (t - 1) * Tdt + 1 : min(t * Tdt, T);
                DL(idx, :) = DLt;
                A(idx, :) = At;
                wws(t, :) = ww + sa;
                wVs(idx, :) = convVW + sa;
                
                % pre-compute updates to \Delta L needed when flipping X_ij
                for i = 1 : M
                    for j = 1 : M
                        for k = 1 : K
                            dDLijk = conv2(conv(flipud(Wt(:, k, i)), Wt(:, k, j)), self.upsamplingFilter);
                            dDL(:, i, j, :, t) = dDL(:, i, j, :, t) + permute(dDLijk(p + 1 : end - p, :), [1 3 4 2]);
                        end
                    end
                end
            end
            
            % greedy search for flips with largest change in posterior
            h = fliplr(self.upsamplingFilter);
            Xn = greedy(sparse(T, M), DL, A, dDL, s, 1 - s(1), T - s(end) + s(1) - 1, h, wws, wVs);
        end
        
        
        function [X, split] = splitTemplates(self, X)
            % Split templates with bimodal amplitude distribution
            
            % Fit mixture of two Gaussians and compare to single Gaussian
            [T, M] = size(X);
            K = 2;
            mu = zeros(M, K);
            sigma = zeros(M, 1);
            prior = zeros(M, K);
            cl = cell(1, K);
            bic = zeros(M, 2);
            for j = 1 : M
                a = full(real(X(X(:, j) > 0, j)));
                [mu(j, :), sigma(j), prior(j, :), cl{j}, bic(j, 2)] = mog1d(a, K, 100);
                % BIC for single Gaussian
                sd = var(a);
                bic(j, 1) = sum((a - mean(a)) .^ 2 / sd + log(2 * pi * sd)) + 2 * log(numel(a));
            end
            rate = full(sum(real(X) > 0, 1))' / (T / self.Fs);
            dprime = abs(diff(mu, [], 2)) ./ sqrt(sigma);
            split = find(bic(:, 1) > bic(:, 2) & ...
                         dprime > self.splitMinDPrime & ...
                         min(prior, [], 2) > self.splitMinPrior & ...
                         rate > self.splitMinRate);
            
            % split clusters and renormalize waveform templates
            for j = split'
                ndx = find(X(:, j));
                i = ndx(cl{j} == 1);
                X(i, j) = X(i, j) / mean(X(i, j));
                i = ndx(cl{j} == 2);
                X(i, end + 1) = X(i, j) / mean(X(i, j)); %#ok
                X(i, j) = 0;
            end
            
            % normalize non-splitted clusters
            for j = setdiff(1 : M, split)
                i = find(X(:, j));
                X(i, j) = X(i, j) / mean(X(i, j));
            end
            
            split = ~isempty(split);
        end
        
        
        function W = pruneWaveforms(self, W)
            % Prune waveforms.
            
            [~, K, M, ~] = size(W);
            
            % smooth with adjacent channels
            x = self.layout.x;
            y = self.layout.y;
            nrm = zeros(K, M);
            N = 0;
            for k = 1 : K
                neighbors = (x - x(k)) .^ 2 + (y - y(k)) .^ 2 < radius ^ 2;
                h = neighbors * (1 - self.pruningCtrWeight) / (sum(neighbors) - 1);
                h(k) = self.pruningCtrWeight;
                nrm(k, :) = sqrt(max(sum(sum(bsxfun(@times, h, W), 2) .^ 2, 1), [], 4));
                N = max(N, sum(neighbors) - 1);
            end
            
            % find contiguous region around maximum above threshold
            for m = 1 : M
                [mx, peak] = max(nrm(:, m));
                active = false(1, K);
                if mx > self.pruningThreshold
                    neighbors = peak;
                    active(neighbors) = true;
                else
                    neighbors = [];
                end
                while ~isempty(neighbors)
                    newNeighbors = false;
                    for k = neighbors
                        newNeighbors = newNeighbors | ...
                            (self.layout.neighbors(k, self.pruningRadius) ...
                                & nrm(:, m)' > self.pruningThreshold & ~active);
                    end
                    neighbors = find(newNeighbors);
                    active(neighbors) = true;
                end
                
                % fill holes (channels below threshold where all neighbors are included)
                for k = 1 : K
                    neighbors = self.layout.neighbors(k, self.pruningRadius);
                    active(k) = active(k) | sum(active(neighbors)) == N;
                end
                
                W(:, ~active, m, :) = 0;
            end
        end
        
        
        function y = interp(self, x, k, shape)
            % Interpolate x using subsample shifts
            %   y = self.interp(x, k) interpolates x, shifting it by k
            %   subsamples (i.e. k / self.subsampling samples).
            
            if nargin < 4
                shape = 'same';
            end
            p = self.upsamplingFactor;
            h = self.upsamplingFilter(:, ceil(p / 2) + k);
            y = convn(x, h, shape);
        end
        
    end
end


function [X, DL, A] = greedy(X, DL, A, dDL, s, offset, T, h, wws, wVs)
    % [X, DL, A] = greedy(X, DL, A, dDL, offset, T) performs a greedy
    %   search for flips with largest change in posterior. We use a divide
    %   & conquer approach, splitting the data at the maximum and
    %   recursively processing each chunk, thus speeding up the maximum
    %   search substantially.
    
    Tmax = 10000;
    if T > Tmax
        % divide & conquer: split at current maximum
        [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
        if ~isnan(i)
            [X, DL, A] = greedy(X, DL, A, dDL, s, offset, i - offset, h, wws, wVs);
            [X, DL, A] = greedy(X, DL, A, dDL, s, i, T - i + offset, h, wws, wVs);
        end
    else
        % regular loop greedily searching maximum
        i = 0;
        while ~isnan(i)
            [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs);
        end
    end
end


function [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, h, wws, wVs)
    % [m, i, j] = findmax(DL, offset, T) finds the maximum change of the
    %   log-posterior (DL) achieved by inserting or removing a spike in the
    %   interval DL(offset + (1 : T), :) and returns indices i and j.
    
    Tdt = ceil(size(X, 1) / size(dDL, 5));
    p = size(dDL, 4);
    [m, ndx] = max(reshape(DL(offset + (1 : T), :), [], 1));
    pad = (size(h, 1) - 1) / 2;
    if m > 0
        i = offset + rem(ndx - 1, T) + 1;
        j = ceil(ndx / T);
        if X(i, j) == 0
            % add spike - subsample
            dl = DL(i + (-pad : pad), j)' * h;
            [~, r] = max(dl);
            a = A(i + (-pad : pad), j)' * h(:, r);
            r = (r - fix(p / 2) - 1) / p;
            % real part: amplitude
            % imaginary part: subsample (> 0 => shift right, < 0 => shift left)
            X(i, j) = a + 1i * r;
        else
            % remove spike
            a = real(X(i, j));
            r = imag(X(i, j));
            X(i, j) = 0;
        end
        DLij = DL(i, j);
        sub = ceil(p / 2) - round(r * p);
        sgn = 2 * (X(i, j) > 0) - 1;
        t = ceil(i / Tdt);
        dA = bsxfun(@times, dDL(:, :, j, sub, t), a * sgn ./ wws(t, :));
        A(i + s, :) = A(i + s, :) - dA;
        DL(i + s, :) = DL(i + s, :) - dA .* (wVs(i + s, :) + a * dDL(:, :, j, sub, t));
        DL(i, j) = -DLij;
    else
        i = NaN;
    end
end

