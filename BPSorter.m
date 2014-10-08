classdef BPSorter < handle
    
    properties %#ok<*PROP,*CPROP>
        TempDir             % temporary folder
        BlockSize           % size of blocks with constant waveform (sec)
        MaxSamples          % max number of samples to use
        HighPass            % highpass cutoff [stop, pass] (Hz)
        NyquistFreq         % Nyquist frequency (Hz)
        DropClusterThresh   % threshold for dropping clusters
        
        % properties used for initialization only
        InitChannelOrder    % channel ordering (x|y|xy|yx)
        InitChannelNum      % number of channels to group
        InitDetectThresh    % multiple of noise SD used for spike detection
        InitExtractWin      % window used for extracting waveforms
        InitNumPC           % number of PCs to keep per channel for sorting
    end
    
    
    properties (SetAccess = private)
        layout
        matfile
        Fs
        N
        K
    end
    
    
    methods
        
        function self = BPSorter(layout, varargin)
            % Constructor for BPSorter class
            
            p = inputParser;
            p.addOptional('TempDir', fullfile(tempdir(), datestr(now(), 'BP_yyyymmdd_HHMMSS')));
            p.addOptional('BlockSize', 60);
            p.addOptional('MaxSamples', 2e7);
            p.addOptional('HighPass', [400 600]);
            p.addOptional('NyquistFreq', 6000);
            p.addOptional('DropClusterThresh', 0.6);
            p.addOptional('InitChannelOrder', 'y');
            p.addOptional('InitChannelNum', 5);
            p.addOptional('InitDetectThresh', 5);
            p.addOptional('InitExtractWin', -8 : 19);
            p.addOptional('InitNumPC', 3);
            p.parse(varargin{:});
            par = fieldnames(p.Results);
            for i = 1 : numel(par)
                self.(par{i}) = p.Results.(par{i});
            end
            
            if isa(layout, 'Layout')
                self.layout = layout;
            else
                self.layout = Layout(layout);
            end
            self.K = self.layout.n;
            
            if ~exist(self.TempDir, 'file')
                mkdir(self.TempDir)
            else
                delete([self.TempDir '/*'])
            end
            
            self.Fs = 2 * self.NyquistFreq;
        end
        
        
        function delete(self)
            % Class destructor
            
            % remove temp directory
            delete(fullfile(self.TempDir, '*'))
            rmdir(self.TempDir)
        end
        
        
        function readData(self, br)
            % Read raw data, downsample and store in local temp file
            
            assert(self.K == getNbChannels(br), ...
                'Dataset and channel layout are incompatible: %d vs. %d channels!', ...
                getNbChannels(br), self.K)
            
            % create memory-mapped Matlab file
            dataFile = fullfile(self.TempDir, 'data.mat');
            save(dataFile, '-v7.3', 'dataFile'); % save something to create the file
            self.matfile = matfile(dataFile, 'writable', true);
            
            % read data, resample, and store to temp file
            Fs = getSamplingRate(self.baseReader);
            fr = filteredReader(self.baseReader, filterFactory.createHighpass(self.HighPass(1), self.HighPass(2), Fs));
            blockSize = round(self.BlockSize * Fs);
            pr = packetReader(fr, 1, 'stride', blockSize);
            [p, q] = rat(self.Fs / Fs);
            nBlocks = length(pr);
            lastBlockSize = ceil((length(fr) - (nBlocks - 1) * blockSize) * p / q);
            newBlockSize = ceil(blockSize * p / q);
            self.N = (nBlocks - 1) * newBlockSize + lastBlockSize;
            h5create(dataFile, '/V', [self.N self.K], 'ChunkSize', [newBlockSize self.K]);
            fprintf('Creating temporary file containing resampled data [%d blocks]\n%s\n', nBlocks, dataFile)
            for i = 1 : nBlocks
                if ~rem(i, 10)
                    fprintf('%d ', i)
                end
                V = toMuV(self.baseReader, resample(pr(i), p, q));
                start = (i - 1) * newBlockSize;
                self.matfile.V(start + (1 : newBlockSize), :) = V;
            end
            fprintf('done\n')
        end
        
        
        function initialize(self)
            % Initialize model
            
            % load subset of the data
            nskip = ceil(self.N / self.MaxSamples);
            if nskip == 1
                V = self.matfile.V; % load full dataset
            else
                blockSize = self.BlockSize * self.Fs;
                nBlocks = fix(self.N / blockSize);
                subBlockSize = round(blockSize / nskip);
                idx = bsxfun(@plus, blockSize * (0 : nBlocks - 1), (1 : subBlockSize)');
                V = self.matfile.V(idx(:), :);
            end
            
            % Create channel groups
            channels = self.layout.channelOrder(self.InitChannelOrder);
            num = self.InitChannelNum;
            idx = bsxfun(@plus, 1 : num, (0 : numel(channels) - num)');
            groups = channels(idx);
            nGroups = size(groups, 1);
            
            % detect and sort spikes in groups
            for i = 1 : nGroups
                Vi = V(:, groups(i, :));
                [s, t] = self.detectSpikes(Vi);
                w = self.extractWaveforms(Vi, s);
                b = self.extractFeatures(w);
                
                % dt needs to be adjusted since we're skipping a fraction of the data
                % drift rate is per ms, so it needs to be adjusted as well
                model = MoKsm('DTmu', par.BlockSize / nskip * 1000, 'DriftRate', par.DriftRate * nskip, ...
                    'ClusterCost', par.ClusterCost, 'Df', par.Df, 'Tolerance', par.Tolerance, 'CovRidge', par.CovRidge);
                model = model.fit(b, t);
                
                % NEED TO DO SOMETHING WITH IT!!
            end
        end
        
    end
    
    
    methods (Access = private)
        
        function [s, t] = detectSpikes(self, V)
            % Detect spikes.
            %   [s, t] = self.detectSpikes(V) detects spikes in V. The
            %   outputs s and t are column vectors of spike times in
            %   samples and ms, respectively. By convention the time of the
            %   zeroth sample is 0 ms.
            
            % detect local minima where at least one channel is above threshold
            noiseSD = median(abs(V)) / 0.6745;
            z = bsxfun(@rdivide, V, noiseSD);
            mz = min(z, [], 2);
            r = sqrt(sum(V .^ 2, 2));
            dr = diff(r);
            s = find(mz(2 : end - 1) < -self.InitDetectThresh & dr(1 : end - 1) > 0 & dr(2 : end) < 0) + 1;
            
            % remove spikes close to boundaries
            s = s(s > self.InitExtractWin(1) & s < size(V, 1) - self.InitExtractWin(end));
            
            % if multiple spikes occur within 1 ms we keep only the largest
            refractory = 1 / 1000 * self.Fs;
            N = numel(s);
            keep = true(N, 1);
            last = 1;
            for i = 2 : N
                if s(i) - s(last) < refractory
                    if r(s(i)) > r(s(last))
                        keep(last) = false;
                        last = i;
                    else
                        keep(i) = false;
                    end
                else
                    last = i;
                end
            end
            s = s(keep);
            t = s / self.Fs * 1000; % convert to real times in ms
        end
        
        
        function w = extractWaveforms(self, V, s)
            % Extract spike waveforms.
            %   w = self.extractWaveforms(V, s) extracts the waveforms at
            %   times s (given in samples) from the filtered signal V using
            %   a fixed window around the times of the spikes. The return
            %   value w is a 3d array of size length(InitExtractWin) x
            %   #spikes x
            %   #channels.
            
            win = self.InitExtractWin;
            idx = bsxfun(@plus, s, win)';
            w = reshape(V(idx, :), [length(win) numel(s) self.InitChannelNum]);
        end
        
        
        function b = extractFeatures(self, w)
            % Extract features for spike sorting.
            %   b = self.extractFeatures(w) extracts features for spike
            %   sorting from the waveforms in w, which is a 3d array of
            %   size length(InitExtractWin) x #spikes x #channels. The
            %   output b is a matrix of size #spikes x #features.
            %
            %   We do PCA on the waveforms of each channel separately and
            %   keep InitNumPC principal components per channel.
            
            [~, n, k] = size(w);
            q = self.InitNumPC;                 % number of components per channel
            w = bsxfun(@minus, w, mean(w, 2));  % center data
            b = zeros(n, k * q);
            for i = 1:k
                C = w(:, :, i) * w(:, :, i)';   % covariance matrix
                [V, ~] = eigs(C, q);            % first q eigenvectors
                b(:, (1 : q) + q * (i - 1)) = w(:, :, i)' * V;
            end
        end
        
    end
    
end
