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
        upsampling  % upsampling factor for spike times
        T           % # samples
        D           % # dimensions
        K           % # channels
        M           % # clusters
    end
    
    methods
        
        function self = BP(varargin)
            % BP constructor
            %   bp = BP('param1', value1, 'param2', value2, ...) constructs
            %   a BP object with the following optional parameters:
            %
            %   window       1x2 vector specifying the time window (ms) to
            %                extract waveforms (peak = 0; default [-0.5 1])
            %   Fs           sampling rate (Hz)
            %   verbose      true|false
            %   tempFiltLen  length of filter for temporal whitening
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('window', [-1 1.5]);
            p.addOptional('Fs', 12000);
            p.addOptional('verbose', false);
            p.addOptional('tempFiltLen', 1.5);
            p.addOptional('upsampling', 5);
            p.parse(varargin{:});
            self.window = p.Results.window;
            self.Fs = p.Results.Fs;
            self.samples = round(self.window(1) * self.Fs / 1000) : round(self.window(2) * self.Fs / 1000);
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
            self.tempFiltLen = p.Results.tempFiltLen;
            self.upsampling = p.Results.upsampling;
        end
        
        
        function self = fit(self, V, X)
            % bp = bp.fit(V, X0) fits the model to waveforms V using the
            %   initial spike sorting results X0.
            %
            %   V    T-by-K     T: number of time bins
            %                   K: number of channels
            %   X0   T-by-M     M: number of clusters
            %        sparse matrix with ones indicating spikes

            q = round(self.tempFiltLen / 1000 * self.Fs);
            while ~converged
                
                % estimate waveforms, whiten data, re-estimate waveforms
                W = BP.estimateWaveforms(V, X, self.samples);
                R = BP.residuals(V, X, W, self.samples);
                Vw = BP.whitenData(V, R, q);
                Ww = BP.estimateWaveforms(Vw, X, self.samples);
                
                % estimate spike trains via binary pursuit
                X = BP.estimateSpikes(Vw, X, Ww, self.samples, self.upsampling);
            end
        end
    end
    
    methods (Static)
        
        function W = estimateWaveforms(V, X, samples)
            % W = estimateWaveforms(V, X, samples) estimates the waveforms
            %   W given the observed voltage V and spike times X. The
            %   vector samples specifies which samples relative to the
            %   spike time should be estimated.
            
            [T, K] = size(V);
            M = size(X, 2);
            D = numel(samples);
            W = zeros(M * D, K);
            for iChan = 1 : K
                [i, j, x] = find(X);
                i = bsxfun(@plus, i, samples);
                i = [i - 1; i]; %#ok
                valid = i > 0 & i <= T;
                j = bsxfun(@plus, (j - 1) * D, 1 : D);
                j = [j; j]; %#ok
                x = repmat([1 - x; x], 1, D);
                MX = sparse(i(valid), j(valid), x(valid), T, D * M);
                W(:, iChan) = (MX' * MX) \ (MX' * V(:, iChan));
            end
            W = reshape(W, [D M K]);
        end
        
        
        function V = whitenData(V, R, q)
            % V = whitenData(V, q) whitens the data V, assuming that the
            %   spatio-temporal covariance separates into a spatial and a
            %   temporal component.
            
            % temporal whitening
            for i = 1 : size(V, 2)
                Lt = toeplitz(xcorr(R(:, i), q));
                Lt = Lt(q + 1 : end, 1 : q + 1);
                w = sqrtm(inv(Lt));
                V(:, i) = conv2(V(:, i), w, 'same');
                R(:, i) = conv2(R(:, i), w, 'same');
            end
            
            % spatial whitening
            V = V * chol(inv(cov(R)))';
        end
        
        
        function V = residuals(V, X, W, samples)
            % R = residuals(V, X, W, samples) computes the residuals by
            %   subtracting the model prediction X * W from the data V. The
            %   vector samples specifies how to center the waveforms for
            %   the convolution.
            
            for i = 1 : size(X, 2)
                spikes = find(X(:, i));
                Wi = permute(W(:, i, :), [1 3 2]);
                for j = 1 : numel(spikes)
                    V(spikes(j) + samples, :) = V(spikes(j) + samples, :) - X(j, i) * Wi;
                    V(spikes(j) + samples - 1, :) = V(spikes(j) + samples - 1, :) - (1 - X(j, i)) * Wi;
                end
            end
        end
        
        
        function Xn = estimateSpikes(V, X, W, samples, up)
            % X = estimateSpikes(V, X, W, samples) estimates the spike
            %   times given the current estimate of the waveforms using
            %   binary pursuit.

            % initialize \Delta L (Eq. 9) assuming X = 0 (no spikes)
            p = sum(X, 1) / size(X, 1);
            gamma = log(1 - p) - log(p);
            ww = sum(sum(W .^ 2, 1), 3) / 2;
            [T, K] = size(V);
            DL = 0;
            for i = 1 : K
                DL = DL + conv2(V(:, i), flipud(W(:, :, i)));
            end
            DL = DL(samples(end) + (1 : T), :);
            DL = bsxfun(@minus, DL, gamma + ww);
            
            % pre-compute updates to \Delta L needed when flipping X_ij
            D = numel(samples);
            s = 1 - D : D - 1;
            M = size(X, 2);
            dDL = zeros((2 * D - 1) * up, M, M);
            for i = 1 : M
                for j = 1 : M
                    for k = 1 : K
                        dDL(:, i, j) = dDL(:, i, j) + conv(upsample(W(:, i, k), up), resample(flipud(W(:, j, k)), up, 1));
                    end
                end
                dDL((find(~s) - 1) * up + (1 : up), i, i) = 0;
            end
            
            % greedy search for flips with largest change in posterior
            win = gausswin(4 * up + 1, 3.5);
            win = win / sum(win) * up;
            Xn = greedy(sparse(T, M), DL, dDL, s, 1 - s(1), T - s(end) + s(1) - 1, up, win);
        end
    end
end


function [X, DL] = greedy(X, DL, dDL, s, offset, T, up, win)
    % [X, DL] = greedy(X, DL, dDL, offset, T) performs a greedy search for
    %   flips with largest change in posterior. We use a divide & conquer
    %   approach, splitting the data at the maximum and recursively
    %   processing each chunk, thus speeding up the maximum search
    %   substantially.
    
    Tmax = 10000;
    if T > Tmax
        % divide & conquer: split at current maximum
        [X, DL, i] = flip(X, DL, dDL, s, offset, T, up, win);
        if ~isnan(i)
            [X, DL] = greedy(X, DL, dDL, s, offset, i - offset, up, win);
            [X, DL] = greedy(X, DL, dDL, s, i, T - i + offset, up, win);
        end
    else
        % regular loop greedily searching maximum
        i = 0;
        while ~isnan(i)
            [X, DL, i] = flip(X, DL, dDL, s, offset, T, up, win);
        end
    end
end


function [X, DL, i] = flip(X, DL, dDL, s, offset, T, up, win)
    % [m, i, j] = findmax(DL, offset, T) finds the maximum change of the
    %   log-posterior (DL) achieved by inserting or removing a spike in the
    %   interval DL(offset + (1 : T), :) and returns indices i and j.
    
    [m, ndx] = max(reshape(DL(offset + (1 : T), :), [], 1));
    if m > 0
        i = offset + rem(ndx - 1, T) + 1;
        j = ceil(ndx / T);
        if ~X(i, j)
            % add spike - subsample
            pad = (numel(win) - 1) / up / 2 + 1;
            dl = upsample(DL(i + (-pad : pad), j), up);
            dl = conv(dl, win, 'valid');
            [~, sub] = max(dl);
            sub = sub - up - 1;
            i = i + 1;
            while sub < 1
                i = i - 1;
                sub = sub + up;
            end
            X(i, j) = sub / up;
        else
            % remove spike
            sub = X(i, j) * up;
            X(i, j) = 0;
        end
        DL(i, j) = -DL(i, j);
        DL(i + s, :) = DL(i + s, :) - (2 * X(i, j) - 1) * dDL(up - sub + 1 : up : end, :, j);
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

