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
        window      % window for extracting waveform: [a b] ms
        samples     % samples corresponding to waveform window
        Fs          % sampling rate
        verbose     % verbose output during fitting?
        tempFiltLen % length of temporal whitening filter (ms)
        upsamplingFactor  % upsampling factor for spike times
        upsamplingFilter        % filter used for subsampling
        upsamplingFilterOrder   % filter order (for subsampling filter)
        pruning     % pruning threshold for subset selection on waveforms
        passband    % passband of continuous input signal
        dt          % time window for tracking waveform drift (sec)
        driftRate   % waveform drift rate (muV, SD per time step)
        sigmaAmpl   % SD of waveform amplitudes (muV)
        D           % # dimensions
    end
    
    methods
        
        function self = BP(varargin)
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
            p.addOptional('pruning', 1);
            p.addOptional('passband', [0.6 5] / 12);
            p.addOptional('dt', 30);
            p.addOptional('driftRate', 0.1);
            p.addOptional('sigmaAmpl', 0.01);
            p.parse(varargin{:});
            self.window = p.Results.window;
            self.Fs = p.Results.Fs;
            self.samples = round(self.window(1) * self.Fs / 1000) : round(self.window(2) * self.Fs / 1000);
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
            self.tempFiltLen = p.Results.tempFiltLen;
            self.upsamplingFactor = p.Results.upsamplingFactor;
            self.pruning = p.Results.pruning;
            self.passband = p.Results.passband;
            self.dt = p.Results.dt;
            self.driftRate = p.Results.driftRate;
            self.sigmaAmpl = p.Results.sigmaAmpl;
            
            % design filter for resampling
            p = self.upsamplingFactor;
            n = 5;
            len = 2 * n * p + 1;
            f = 1 / p;
            h = p * firls(len - 1, [0 f f 1], [1 1 0 0])' .* kaiser(len, 5);
            self.upsamplingFilter = [zeros(p, 1); h; zeros(p, 1)];
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
                iter = 3;
            end
            for i = 1 : iter
                W = self.estimateWaveforms(V, X);
                R = self.residuals(V, X, W);
                Vw = self.whitenData(V, R);
                Ww = self.estimateWaveforms(Vw, X, self.pruning);
                X = self.estimateSpikes(Vw, X, Ww);
            end
            W = self.estimateWaveforms(V, X);
        end
        
        
        function W = estimateWaveforms(self, V, X, pruning)
            % Estimate waveform templates given spike times.
            %   W = self.estimateWaveforms(V, X) estimates the waveforms W
            %   given the observed voltage V and the current estimate of
            %   the spike times X.
            %
            %   W = self.estimateWaveforms(V, X, pruning) applies subset
            %   selection on the waveforms using the given pruning factor
            %   (multiples of the noise amplitude).
            
            [T, K] = size(V);
            M = size(X, 2);
            D = numel(self.samples);
            Tdt = self.dt * self.Fs;
            Ndt = ceil(T / Tdt);
            
            % Pre-compute convolution matrix: MX * W = conv(X, W)
            [i, j, x] = find(X);
            x = x - 1;
            d = 2 * (x > 0) - 1;
            i = [i; i + d];
            i = bsxfun(@plus, i, self.samples);
            valid = find(i > 0 & i <= T);
            j = bsxfun(@plus, (j - 1) * D, 1 : D);
            j = [j; j];
            x = repmat([1 - abs(x); abs(x)], 1, D);
            
            [i, order] = sort(i(valid));
            j = j(valid(order));
            x = x(valid(order));
            
            borders = zeros(1, Ndt + 1);
            chunk = 1;
            k = 1;
            while chunk < Ndt && k <= numel(i)
                if i(k) > chunk * Tdt
                    n = ceil(i(k) / Tdt) - chunk;
                    borders(chunk + (1 : n)) = k - 1;
                    chunk = chunk + n;
                end
                k = k + 1;
            end
            borders(chunk + 1 : end) = numel(i);

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
            
            % subset selection of waveforms
            if nargin > 3 && pruning > 0
                W(:, sqrt(sum(W .^ 2, 1)) < pruning) = 0;
            end
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
                    r = X(spikes(j), i) - 1;
                    s = sign(r);
                    t = ceil(spikes(j) / Tdt);
                    samples = spikes(j) + self.samples;
                    valid = samples > 0 & samples < T;
                    V(samples(valid), :) = V(samples(valid), :) - (1 - abs(r)) * W(valid, :, i, t);
                    samples = samples + s;
                    valid = samples > 0 & samples < T;
                    V(samples(valid), :) = V(samples(valid), :) - abs(r) * W(valid, :, i, t);
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
            M = size(X, 2);
            dDL = zeros((2 * D) * p, M, M, Ndt);
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
                            dDL(:, i, j, t) = dDL(:, i, j, t) + conv(upsample([0; flipud(Wt(:, k, i))], p), resample(Wt(:, k, j), p, 1));
                        end
                    end
                end
            end
            
            % greedy search for flips with largest change in posterior
            win = gausswin(4 * p + 1, 3.5);
            win = win / sum(win) * p;
            Xn = greedy(sparse(T, M), DL, A, dDL, s, 1 - s(1), T - s(end) + s(1) - 1, p, win, wws, wVs);
        end
        
        
        function y = interp(self, x, k, shape)
            % Interpolate x using subsample shifts
            %   y = self.interp(x, k) interpolates x, shifting it by k
            %   subsamples (i.e. k / self.subsampling samples).
            
            if nargin < 4
                shape = 'same';
            end
            p = self.upsamplingFactor;
            n = numel(self.upsamplingFilter) - 2 * p;
            h = self.upsamplingFilter((p + k) + (1 : p : n));
            y = convn(x, h, shape);
        end
        
    end
end


function [X, DL, A] = greedy(X, DL, A, dDL, s, offset, T, up, win, wws, wVs)
    % [X, DL, A] = greedy(X, DL, A, dDL, offset, T) performs a greedy
    %   search for flips with largest change in posterior. We use a divide
    %   & conquer approach, splitting the data at the maximum and
    %   recursively processing each chunk, thus speeding up the maximum
    %   search substantially.
    
    Tmax = 10000;
    if T > Tmax
        % divide & conquer: split at current maximum
        [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, up, win, wws, wVs);
        if ~isnan(i)
            [X, DL, A] = greedy(X, DL, A, dDL, s, offset, i - offset, up, win, wws, wVs);
            [X, DL, A] = greedy(X, DL, A, dDL, s, i, T - i + offset, up, win, wws, wVs);
        end
    else
        % regular loop greedily searching maximum
        i = 0;
        while ~isnan(i)
            [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, up, win, wws, wVs);
        end
    end
end


function [X, DL, A, i] = flip(X, DL, A, dDL, s, offset, T, up, win, wws, wVs)
    % [m, i, j] = findmax(DL, offset, T) finds the maximum change of the
    %   log-posterior (DL) achieved by inserting or removing a spike in the
    %   interval DL(offset + (1 : T), :) and returns indices i and j.
    
    Tdt = ceil(size(X, 1) / size(dDL, 4));
    ns = numel(s) - 1;
    [m, ndx] = max(reshape(DL(offset + (1 : T), :), [], 1));
    if m > 0
        i = offset + rem(ndx - 1, T) + 1;
        j = ceil(ndx / T);
        if X(i, j) == 0
            % add spike - subsample
            pad = (numel(win) - 1) / up / 2 + 1;
            dl = upsample(DL(i + (-pad : pad), j), up);
            dl = conv(dl(ceil(up / 2) + 1 : end - ceil(up / 2)), win, 'valid');
            [~, r] = max(dl);
            a = upsample(A(i + (-pad : pad), j), up);
            a = conv(a(ceil(up / 2) + 1 : end - ceil(up / 2)), win, 'valid');
            a = a(r);
            r = (r - fix(up / 2) - 1) / up;
            % real aprt: subsample (> 1 => shift right, < 1 => shift left)
            % imaginary part: amplitude
            X(i, j) = (1 + r) + 1i * a;
        else
            % remove spike
            r = real(X(i, j) - 1);
            a = imag(X(i, j));
            X(i, j) = 0;
        end
        DLij = DL(i, j);
        sub = up + 1 - round(r * up);
        sgn = 2 * (X(i, j) > 0) - 1;
        t = ceil(i / Tdt);
        dA = bsxfun(@times, dDL(sub + (0 : ns) * up, :, j, t), a * sgn ./ wws(t, :));
        A(i + s, :) = A(i + s, :) - dA;
        DL(i + s, :) = DL(i + s, :) - dA .* (wVs(i + s, :) + a * dDL(sub + (0 : ns) * up, :, j, t));
        DL(i, j) = -DLij;
    else
        i = NaN;
    end
end


function y = upsample(x, k)
    % y = upsample(x, up) up-samples vector x k times by inserting zeros.
    
    n = numel(x);
    y = zeros((n - 1) * k + 1, 1);
    y((0 : n - 1) * k + 1) = x;
end

