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
        N           % # samples
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
            p.parse(varargin{:});
            self.window = p.Results.window;
            self.Fs = p.Results.Fs;
            self.samples = round(self.window(1) * self.Fs / 1000) : round(self.window(2) * self.Fs / 1000);
            self.D = numel(self.samples);
            self.verbose = p.Results.verbose;
        end
        
    end
    
    methods (Static)
        
        function W = estimateWaveforms(V, X, samples)
            % W = estimateWaveforms(V, X, samples) estimates the waveforms
            %   W given the observed voltage V and spike times X. The
            %   vector samples specifies which samples relative to the
            %   spike time should be estimated.
            
            [N, K] = size(V);
            M = size(X, 2);
            D = numel(samples);
            W = zeros(M * D, K);
            for iChan = 1 : K
                MX = sparse(N, D * M);
                for iSample = 1 : D
                    index = (1 : N) + samples(iSample);
                    valid = index > 0 & index <= N;
                    MX(index(valid), iSample + (0 : D : end - D)) = X(valid, :); %#ok<SPRIX>
                end
                W(:, iChan) = (MX' * MX) \ (MX' * V(:, iChan));
            end
            W = reshape(W, [D M K]);
        end
        
    end
end
