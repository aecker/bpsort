% Spike sorting using binary pursuit.
%
% Implements the algorithm described in the following paper: 
% Pillow, Shlens, Chichilnisky & Simoncelli (2013): A Model-Based Spike
% Sorting Algorithm for Removing Correlation Artifacts in Multi-Neuron
% Recordings. PLoS ONE 8, e62123.
%
% AE 2014-08-07

classdef BP < handle
    properties %#ok<*PROP>
        layout              % geometrical layout of the electrode (see Layout class)
        samples             % samples corresponding to waveform window
        Fs                  % sampling rate
        verbose             % verbose output during fitting?
        logging             % log progress into file?
        logFile             % name of log file (default: yyyymmdd_HHMMSS.log)
        tempFiltLen         % length of temporal whitening filter (ms)
        upsamplingFactor    % upsampling factor for spike times
        upsamplingFilter    % filter used for subsampling
        upsamplingFilterOrder   % filter order (for subsampling filter)
        passband            % passband of continuous input signal
        dt                  % time window for tracking waveform drift (sec)
        driftRate           % waveform drift rate (muV, SD per time step)
        sigmaAmpl           % SD of waveform amplitudes (muV)
        splitMinDPrime      % Min d' on aomplitudes for splitting a cluster
        splitMinPrior       % Min prior prob ob second component for splitting
        splitMinRate        % Min firing rate for splitting
        pruningRadius       % radius for smoothing before pruning
        pruningCtrWeight    % center weight of smoothing filter
        pruningThreshold    % pruning threshold
        mergeThreshold      % merging threshold (maximal cross-correlation)
        waveformBasis       % basis vector for waveforms

        K                   % # channels
        D                   % # samples/waveform/channel
        E                   % # basis functions for waveforms/channel
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
            p.addOptional('samples', -11 : 24);
            p.addOptional('Fs', 12000);
            p.addOptional('verbose', false);
            p.addOptional('logging', false);
            p.addOptional('tempFiltLen', 0.5);
            p.addOptional('upsamplingFactor', 5, @(p) assert(mod(p, 2) == 1, 'Upsampling factor must be odd!'));
            p.addOptional('passband', [0.6 5] / 12);
            p.addOptional('dt', 60);
            p.addOptional('driftRate', 0.01);
            p.addOptional('sigmaAmpl', 0.05);
            p.addOptional('splitMinDPrime', 1);
            p.addOptional('splitMinPrior', 0.05);
            p.addOptional('splitMinRate', 0.1);
            p.addOptional('pruningRadius', 1);
            p.addOptional('pruningCtrWeight', 0.7);
            p.addOptional('pruningThreshold', 2);
            p.addOptional('mergeThreshold', 0.95);
            p.addOptional('waveformBasis', []);
            p.parse(varargin{:});
            self.samples = p.Results.samples;
            self.Fs = p.Results.Fs;
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
            self.logging = p.Results.logging;
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
            self.mergeThreshold = p.Results.mergeThreshold;
            
            % store or read electrode layout
            if isa(layout, 'Layout')
                self.layout = layout;
            else
                self.layout = Layout(layout);
            end
            self.K = self.layout.n;
            
            % normalize waveform basis (W = BU)
            B = p.Results.waveformBasis;
            assert(isempty(B) || size(B, 1) == self.D, 'Waveform basis must be of dimensionality %d!', self.D)
            self.waveformBasis = bsxfun(@rdivide, B, sqrt(sum(B .* B, 1)));
            self.E = size(B, 2);
            
            % design filter for resampling
            p = self.upsamplingFactor;
            n = 5;
            len = 2 * n * p + 1;
            f = 1 / p;
            h = p * firls(len - 1, [0 f f 1], [1 1 0 0])' .* kaiser(len, 5);
            h = [zeros(fix(p / 2), 1); h; zeros(fix(p / 2), 1)];
            self.upsamplingFilter = reshape(h, p, 2 * n + 1)';
            self.upsamplingFilterOrder = n;
            
            % determine file name for log file
            if self.logging
                p = mfilename('fullpath');
                ndx = find(p == filesep, 1, 'last');
                self.logFile = [p(1 : ndx), 'logs', filesep, datestr(now, 'yyyymmdd_HHMMSS'), '.log'];
            end
        end
        
        
        function [X, U] = fit(self, V, X, iter)
            % Fit model (i.e. estimate waveform templates and spike times).
            %   [X, U] = self.fit(V, X0) fits the model to voltage trace V
            %   using the initial spike times X0.
            %
            %   [X, U] = self.fit(V, X0, iter) uses the specified number of
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
            %   U       Array of waveform coefficients
            %           E-by-K-by-M    E: number of basis functions/samples
            %                          K: number of channels
            %                          M: number of neurons

            self.log('Starting to fit model\n')
            t = now;
            if nargin < 4
                iter = 2;
            end
            
            % initial estimate of waveforms in non-whitened, whitening
            U = self.estimateWaveforms(V, X);
            R = self.residuals(V, X, U);
            Vw = self.whitenData(V, R);
            
            split = true;
            doneSplitMerge = false;
            priors = sum(X > 0, 1) / size(X, 1);
            i = 0;
            M = 0;
            while i < iter || ~doneSplitMerge
                
                % estimate waveforms
                Uw = self.estimateWaveforms(Vw, X);
                
                % merge templates that are too similar
                if ~doneSplitMerge
                    [Uw, priors, merged] = self.mergeTemplates(Uw, priors);
                end
                
                % stop merging when number of templates decreases compared
                % to previous iteration
                if numel(priors) < M || (~split && ~merged)
                    doneSplitMerge = true;
                else
                    M = numel(priors);
                end

                % prune waveforms and estimate spikes
                Uw = self.pruneWaveforms(Uw);
                [X, priors] = self.estimateSpikes(Vw, Uw, priors);
                
                % split templates with bimodal amplitude distribution
                if ~doneSplitMerge
                    [X, priors, split] = self.splitTemplates(X, priors);
                else
                    i = i + 1;
                end
                
                self.log('\n')
            end
            
            % Re-estimate non-whitened waveforms and apply the same pruning
            % as to whitened waveforms
            U = self.estimateWaveforms(V, X);
            nnz = max(sum(abs(Uw), 1), [], 4) > 1e-6;
            U = bsxfun(@times, U, nnz);
            
            % Order templates spatially
            [U, X] = self.orderTemplates(U, X, priors, 'yx');
            
            self.log('Done fitting model [%.0fs]\n\n', (now - t) * 24 * 60 * 60)
        end
        
        
        function U = estimateWaveforms(self, V, X)
            % Estimate waveform templates given spike times.
            %   U = self.estimateWaveforms(V, X) estimates the waveform
            %   coefficients U given the observed voltage V and the current
            %   estimate of the spike times X.
            
            self.log(false, 'Estimating waveforms... ')
            [T, K] = size(V);
            M = size(X, 2);
            D = numel(self.samples);
            Tdt = round(self.dt * self.Fs);
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
            
            B = self.waveformBasis;
            if isempty(B)
                E = D;
            else
                E = size(B, 2);
            end
            U = zeros(E * M, K, Ndt);
            Q = eye(E * M) * self.dt * self.driftRate;
            
            % Pre-compute MX' * MX
            if K > 1
                MX = cell(1, Ndt);
                BMXprod = zeros(E * M, E * M, Ndt);
                for t = 2 : Ndt
                    idx = borders(t) + 1 : borders(t + 1);
                    MX{t} = sparse(i(idx) - (t - 1) * Tdt, j(idx), x(idx), Tdt, D * M);
                    BMXprod(:, :, t) = getBMXprod(MX{t}, B);
                end
            end
            
            % Initialize
            MX1 = sparse(i(1 : borders(2)), j(1 : borders(2)), x(1 : borders(2)), Tdt, D * M);
            MX1V = MX1' * V(1 : Tdt, :);
            if isempty(B)
                BMX1V = MX1V;
            else
                BMX1V = zeros(E * M, K);
                for m = 1 : M
                    iD = (m - 1) * D + (1 : D);
                    iE = (m - 1) * E + (1 : E);
                    BMX1V(iE, :) = B' * MX1V(iD, :);
                end
            end
            
            % using pinv() instead of \ because MXprod can be rank-
            % deficient if there are no or only few spikes for some neurons
            U(:, :, 1) = pinv(getBMXprod(MX1, B)) * BMX1V;
            
            % Initialize state covariance
            n = full(sum(MX1, 1));
            P = zeros(E * M, E * M, Ndt);
            P1 = diag(1 ./ (n + ~n));
            if ~isempty(B)
                for m = 1 : M
                    iD = (m - 1) * D + (1 : D);
                    iE = (m - 1) * E + (1 : E);
                    P(iE, iE, 1) = B' * P1(iD, iD) * B;
                end
            else
                P(:, :, 1) = P1;
            end
            
            % Go through all channels
            for k = 1 : K
                
                % Forward pass
                Pti = zeros(E * M, E * M, Ndt);
                I = eye(E * M);
                for t = 2 : Ndt
                    
                    % Predict
                    Pt = P(:, :, t - 1) + Q;
                    Pti(:, :, t) = inv(Pt);
                    Ut = U(:, k, t - 1);
                    
                    % Update
                    if K > 1
                        MXt = MX{t};
                        BMXp = BMXprod(:, :, t);
                    else
                        idx = borders(t) + 1 : borders(t + 1);
                        MXt = sparse(i(idx) - (t - 1) * Tdt, j(idx), x(idx), Tdt, D * M);
                        BMXp = getBMXprod(MXt, B);
                    end
                    Kp = Pt * (I - BMXp / (Pti(:, :, t) + BMXp)); % Kalman gain (K = Kp * MX)
                    KpBMXp = Kp * BMXp;
                    tt = (t - 1) * Tdt + (1 : Tdt);
                    MXtV = MXt' * V(tt, k);
                    if isempty(B)
                        BMXtV = MXtV;
                    else
                        BMXtV = zeros(E * M, 1);
                        for m = 1 : M
                            iD = (m - 1) * D + (1 : D);
                            iE = (m - 1) * E + (1 : E);
                            BMXtV(iE) = B' * MXtV(iD);
                        end
                    end
                    U(:, k, t) = Ut + Kp * BMXtV - KpBMXp * Ut;
                    P(:, :, t) = (I - KpBMXp) * Pt;
                end
                
                % Backward pass
                for t = Ndt - 1 : -1 : 1
                    Ct = P(:, :, t) * Pti(:, :, t + 1);
                    U(:, k, t) = U(:, k, t) + Ct * (U(:, k, t + 1) - U(:, k, t));
                end
            end
            
            % Re-organize waveforms by cluster
            U = reshape(U, [E M K Ndt]);
            U = permute(U, [1 3 2 4]);
            
            self.log(true)
        end
        
        
        function [V, temporal, spatial] = whitenData(self, V, R)
            % Whiten data.
            %   V = self.whitenData(V, R) whitens the data V, assuming
            %   that the spatio-temporal covariance separates into a
            %   spatial and a temporal component. Whitening filters are
            %   estimated from the residuals R.

            self.log(false, 'Whitening data... ')
            
            % determine frequencies outside the passband to avoid
            % amplification of those frequencies
            q = round(self.tempFiltLen / 1000 * self.Fs);
            k = 4 * q + 1;
            F = linspace(0, 2, k + 1);
            F = F(1 : end - 1);
            high = find(F > self.passband(2) & F < 2 - self.passband(2));
            low = F < self.passband(1) | F > 2 - self.passband(1);
            Q = dftmtx(k);
            
            % temporal whitening
            temporal = zeros(2 * q + 1, self.K);
            for i = 1 : self.K
                
                % construct filter for temporal whitening
                c = xcorr(R(:, i), 2 * q, 'coeff');
                c = ifftshift(c);
                ci = 1./ abs(fft(c));
                if ~isempty(high)
                    ci(high) = ci(high(1) - 1);
                end
                ci(low) = 0;
                w = real(Q * (sqrt(ci) .* Q(2 * q + 1, :)') / k);
                w = w(q + 1 : end - q);
                temporal(:, i) = w;

                % apply temporal whitening filter
                V(:, i) = conv(V(:, i), w, 'same');
                R(:, i) = conv(R(:, i), w, 'same');
            end
            
            % spatial whitening
            spatial = chol(inv(cov(R)))';
            V = V * spatial;
            
            self.log(true)
        end
        
        
        function W = waveforms(self, U)
            % Return waveform templates given the coefficients.
            %   W = bp.waveforms(U) returns the waveform templates W given
            %   the coefficients U.
            
            B = self.waveformBasis;
            if isempty(B)
                W = U;
            else
                [~, K, M, T] = size(U);
                W = zeros([self.D, K, M, T]);
                for m = 1 : M
                    for t = 1 : T
                        W(:, :, m, t) = B * U(:, :, m, t);
                    end
                end
            end
        end
        
        
        function V = residuals(self, V, X, U)
            % Compute residuals by subtracting waveform templates.
            %   R = self.residuals(V, X, U) computes the residuals by
            %   subtracting the model prediction X * W from the data V.
            
            self.log(false, 'Computing residuals... ')
            W = self.waveforms(U);
            T = size(V, 1);
            Tdt = round(self.dt * self.Fs);
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
            self.log(true)
        end
        
        
        function [X, priors] = estimateSpikes(self, V, U, priors)
            % Estimate spike times given waveform templates.
            %   [X, priors] = self.estimateSpikes(V, U, priors) estimates
            %   the spike times given the current estimate of the waveforms
            %   using binary pursuit.

            self.log(false, 'Estimating spike times... ')
            [T, K] = size(V);
            M = numel(priors);
            Tdt = round(self.dt * self.Fs);
            Ndt = ceil(T / Tdt);
            DL = zeros(T, M);
            A = zeros(T, M);
            wws = zeros(Ndt, M);
            wVs = zeros(T, M);
            p = self.upsamplingFactor;
            B = self.waveformBasis;
            E = size(B, 2);
            D = self.D;
            s = 1 - D : D - 1;
            dDL = zeros(2 * D - 1, M, M, p, Ndt);
            
            % pre-compute convolutions of all basis functions
            if ~isempty(B)
                convBB = zeros(E, E, 2 * D - 1, p);
                for i = 1 : E
                    for j = 1 : E
                        t = conv2(conv(flipud(B(:, i)), B(:, j)), self.upsamplingFilter);
                        convBB(i, j, :, :) = t(p + 1 : end - p, :);
                    end
                end
            end
            
            for t = 1 : Ndt
                Ut = U(:, :, :, t);
                
                % initialize \Delta L (Eq. 9) assuming X = 0 (no spikes)
                gamma = log(1 - priors) - log(priors);
                ww = permute(sum(sum(Ut .* Ut, 1), 2), [1 3 2]);
                convVW = 0;
                if isempty(B)
                    for k = 1 : K
                        Uk = permute(Ut(:, k, :), [1 3 2]);
                        Vk = V(max(1, (t - 1) * Tdt - self.samples(end) + 1) : min(T, t * Tdt - self.samples(1)), k);
                        convVWk = conv2(Vk, flipud(Uk)); % O(KNDM)
                        first = (1 + (t > 1)) * self.samples(end) + 1;
                        last = size(convVWk, 1) + (1 + (t < Ndt)) * self.samples(1);
                        convVW = convVW + convVWk(first : last, :); % O(KNM)
                    end
                else
                    % for large M and K > E loop over e is faster than over k
                    Vt = V(max(1, (t - 1) * Tdt - self.samples(end) + 1) : min(T, t * Tdt - self.samples(1)), :);
                    for e = 1 : E
                        Ue = permute(Ut(e, :, :), [2 3 1]);
                        convVeB = conv2(Vt, flipud(B(:, e))); % O(KNED)
                        first = (1 + (t > 1)) * self.samples(end) + 1;
                        last = size(convVeB, 1) + (1 + (t < Ndt)) * self.samples(1);
                        convVW = convVW + convVeB(first : last, :) * Ue; % O(KNEM)
                    end
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
                        if isempty(B)
                            for k = 1 : K
                                dDLijk = conv2(conv(flipud(Ut(:, k, i)), Ut(:, k, j)), self.upsamplingFilter);
                                dDL(:, i, j, :, t) = dDL(:, i, j, :, t) + permute(dDLijk(p + 1 : end - p, :), [1 3 4 2]);
                            end
                        else
                            uu = Ut(:, :, i) * Ut(:, :, j)';
                            dDLijk = sum(sum(bsxfun(@times, convBB, uu), 1), 2);
                            dDL(:, i, j, :, t) = permute(dDLijk, [3 1 4 2]);
                        end
                    end
                end
            end
            
            % greedy search for flips with largest change in posterior
            h = fliplr(self.upsamplingFilter);
            X = greedy(sparse(T, M), DL, A, dDL, s, 1 - s(1), T - s(end) + s(1) - 1, h, wws, wVs);
            priors = sum(X > 0, 1) / T;
            
            self.log(true)
        end
        
        
        function [U, priors, merged] = mergeTemplates(self, U, priors)
            % Merge templates with similar waveforms.
            %   [W, priors, merged] = bp.mergeTemplates(W, priors) merges
            %   all templates in W whose maximal cross-correlation is
            %   greater than bp.mergeThreshold times the squared norm of
            %   the larger waveform.
            
            M = size(U, 3);
            self.log(false, 'Merging templates: %d -> ', M)
            p = self.upsamplingFactor;
            h = self.upsamplingFilter;
            W = self.waveforms(U);
            W = permute(W, [1 2 4 3]);
            W = reshape(W, [], M);
            nrm = sum(W .* W, 1);
            lag = 2;
            xcp = zeros(p, 2 * lag + 1);
            merged = false;
            i = 1;
            while i < M
                XC = zeros(1, M - i);
                for j = i + 1 : M
                    xc = xcorr(W(:, i), W(:, j), self.upsamplingFilterOrder + lag);
                    for k = 1 : 2 * lag + 1
                        xcp(:, k) = xc(2 * lag + 2 - k : end - k + 1)' * h;
                    end
                    XC(j - i) = max(xcp(:));
                end
                XC = XC ./ max(nrm(i), nrm(i + 1 : M));
                merge = i + [0, find(XC > self.mergeThreshold)];
                if numel(merge) > 1
                    [~, ndx] = max(priors(merge));
                    U(:, :, i, :) = U(:, :, merge(ndx), :);
                    U(:, :, merge(2 : end), :) = [];
                    priors(i) = sum(priors(merge));
                    priors(merge(2 : end)) = [];
                    nrm(i) = nrm(merge(ndx));
                    nrm(merge(2 : end)) = [];
                    M = numel(priors);
                    merged = true;
                end
                i = i + 1;
            end
            self.log('%d ', M)
            self.log(true)
        end
        
        
        function [X, priors, split] = splitTemplates(self, X, priors)
            % Split templates with bimodal amplitude distribution
            
            self.log(false, 'Splitting templates: %d -> ', numel(priors))
            
            % First remove unused templates
            n = full(sum(X > 0, 1));
            X(:, ~n) = [];
            priors(~n) = [];
            
            % Fit mixture of two Gaussians and compare to single Gaussian
            [T, M] = size(X);
            K = 2;
            mu = zeros(M, K);
            sigma = zeros(M, 1);
            prior = zeros(M, K);
            cl = cell(1, K);
            bic = zeros(M, 2);
            rate = n(n > 0)' / (T / self.Fs);
            for j = 1 : M
                if rate(j) > self.splitMinRate
                    a = full(real(X(X(:, j) > 0, j)));
                    [mu(j, :), sigma(j), prior(j, :), cl{j}, bic(j, 2)] = mog1d(a, K, 100);
                    
                    % BIC for single (left-truncated) Gaussian
                    Z = @(m, s) max(normcdf(-2), 1 - normcdf(min(a), m, s));
                    logpdf = @(x, m, s) -0.5 * ((x - m) .^ 2 / s ^ 2 + log(2 * pi)) - log(s) - log(Z(m, s));
                    p = mle(a, 'logpdf', logpdf, 'start', [mean(a) std(a)]);
                    bic(j, 1) = -2 * sum(logpdf(a, p(1), p(2))) + 3 * log(numel(a));
                end
            end
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
                pj = priors(j);
                priors(j) = pj * prior(1);
                i = ndx(cl{j} == 2);
                X(i, end + 1) = X(i, j) / mean(X(i, j)); %#ok
                X(i, j) = 0;
                priors(end + 1) = pj * prior(2); %#ok
            end
            
            % normalize non-splitted clusters
            for j = setdiff(1 : M, split)
                i = find(X(:, j));
                X(i, j) = X(i, j) / mean(X(i, j));
            end
            
            split = ~isempty(split);
            
            self.log('%d ', numel(priors))
            self.log(true)
        end
        
        
        function U = pruneWaveforms(self, U)
            % Prune waveforms.
            
            self.log(false, 'Pruning waveforms... ')
            [~, K, M, ~] = size(U);
            
            % smooth with adjacent channels
            nrm = zeros(K, M);
            N = 0;
            for k = 1 : K
                neighbors = self.layout.neighbors(k, self.pruningRadius);
                h = zeros(1, K);
                h(neighbors) = (1 - self.pruningCtrWeight) / numel(neighbors);
                h(k) = self.pruningCtrWeight;
                nrm(k, :) = sqrt(max(sum(sum(bsxfun(@times, h, U), 2) .^ 2, 1), [], 4));
                N = max(N, numel(neighbors));
            end
            
            % find contiguous region around maximum above threshold
            for m = 1 : M
                [mx, peak] = max(nrm(:, m));
                active = false(K, 1);
                if mx > self.pruningThreshold
                    neighbors = peak;
                    active(neighbors) = true;
                else
                    neighbors = [];
                end
                while ~isempty(neighbors)
                    newNeighbors = false;
                    for k = neighbors(:)'
                        newNeighbors = newNeighbors | ...
                            (self.layout.isNeighbor(1 : K, k, self.pruningRadius) ...
                                & nrm(:, m) > self.pruningThreshold & ~active);
                    end
                    neighbors = find(newNeighbors);
                    active(neighbors) = true;
                end
                
                % fill holes (channels below threshold where all neighbors are included)
                for k = 1 : K
                    neighbors = self.layout.neighbors(k, self.pruningRadius);
                    active(k) = active(k) | sum(active(neighbors)) == N;
                end
                
                U(:, ~active, m, :) = 0;
            end
            self.log(true)
        end
        
        
        function [U, X, priors, order] = orderTemplates(self, U, X, priors, orderBy)
            % Order waveform templates spatially.
            %   [U, X, priors] = orderTemplates(self, U, X, priors, 'y')
            %   orders the waveform templates spatially by th y-location of
            %   the channel with maximum energy.
            
            M = numel(priors);
            order = self.layout.channelOrder(orderBy);
            mag = zeros(1, M);
            peak = zeros(1, M);
            for m = 1 : M
                Ui = mean(U(:, order, m, :), 4);
                [mag(m), peak(m)] = max(sum(Ui .* Ui, 1));
            end
            [~, order] = sort(peak * 1e6 - mag);
            U = U(:, :, order, :);
            X = X(:, order);
            priors = priors(order);
        end
        
        
    end
    
    
    
    methods (Access = protected)
        
        function log(self, varargin)
            
            % first input numeric: 0 = starting / 1 = done with step
            if islogical(varargin{1})
                if ~varargin{1}
                    varargin(1) = [];
                    tic
                else
                    varargin{1} = 'done [%.1fs]\n';
                    varargin{2} = toc;
                end
            end
                    
            % write to log file?
            if self.logging
                fid = fopen(self.logFile, 'a');
                assert(fid > 0, 'Failed to open log file %s!', self.logFile)
                fprintf(fid, varargin{:});
                fclose(fid);
            end

            % print to command line?
            if self.verbose
                fprintf(varargin{:})
            end
        end
        
    end
end


function BMXprod = getBMXprod(MX, B)
% Compute B' * MX' * MX * B efficiently.

    MXp = MX' * MX;
    if ~isempty(B)
        [D, E] = size(B);
        M = size(MX, 2) / D;
        BMXprod = zeros(E * M);
        for mi = 1 : M
            iD = (mi - 1) * D + (1 : D);
            iE = (mi - 1) * E + (1 : E);
            for mj = 1 : M
                jD = (mj - 1) * D + (1 : D);
                jE = (mj - 1) * E + (1 : E);
                BMXprod(iE, jE) = B' * MXp(iD, jD) * B;
            end
        end
    else
        BMXprod = MXp;
    end
end
