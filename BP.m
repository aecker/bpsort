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
            %   window    1x2 vector specifying the time window (ms) to
            %             extract waveforms (zero = peak; default [-0.5 1])
            %   Fs        sampling rate (Hz)
            %   verbose   true|false
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('window', [-0.5 1.1]);
            p.addOptional('Fs', 12000);
            p.addOptional('verbose', false);
            p.addOptional('tempFiltLen', 8);
            p.parse(varargin{:});
            self.window = p.Results.window;
            self.Fs = p.Results.Fs;
            self.samples = round(self.window(1) * self.Fs / 1000) : round(self.window(2) * self.Fs / 1000);
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
            self.tempFiltLen = p.Results.tempFiltLen;
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
                X = BP.estimateSpikes(Vw, X, Ww, self.samples);
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
                MX = sparse(T, D * M);
                for iSample = 1 : D
                    index = (1 : T) + samples(iSample);
                    valid = index > 0 & index <= T;
                    MX(index(valid), iSample + (0 : D : end - D)) = X(valid, :); %#ok<SPRIX>
                end
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
                for j = 1 : numel(spikes)
                    V(spikes(j) + samples, :) = V(spikes(j) + samples, :) - permute(W(:, i, :), [1 3 2]);
                end
            end
        end
        
    end
end
