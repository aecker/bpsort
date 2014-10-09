classdef BPSorter < handle
    
    properties %#ok<*PROP,*CPROP>
        Debug               % debug mode (true|false)
        TempDir             % temporary folder
        BlockSize           % size of blocks with constant waveform (sec)
        MaxSamples          % max number of samples to use
        HighPass            % highpass cutoff [stop, pass] (Hz)
        NyquistFreq         % Nyquist frequency (Hz)
        
        % properties used for initialization only
        InitChannelOrder    % channel ordering (x|y|xy|yx)
        InitNumChannels     % number of channels to group
        InitDetectThresh    % multiple of noise SD used for spike detection
        InitExtractWin      % window used for extracting waveforms
        InitNumPC           % number of PCs to keep per channel for sorting
        InitDropClusterThresh   % threshold for dropping clusters
        InitOverlapTime     % minimum distance between two spikes (ms)
        
        % parameters for initial spike sorting (see MoKsm)
        InitSortDf          % degrees of freedom
        InitSortClusterCost % penalty for adding cluster
        InitSortDriftRate   % drift rate for mean waveform (Kalman filter)
        InitSortTolerance   % convergence criterion
        InitSortCovRidge    % ridge on covariance matrices (regularization)
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
            p.addOptional('Debug', false);
            p.addOptional('TempDir', fullfile(tempdir(), datestr(now(), 'BP_yyyymmdd_HHMMSS')));
            p.addOptional('BlockSize', 60);
            p.addOptional('MaxSamples', 2e7);
            p.addOptional('HighPass', [400 600]);
            p.addOptional('NyquistFreq', 6000);
            p.addOptional('InitChannelOrder', 'y');
            p.addOptional('InitNumChannels', 5);
            p.addOptional('InitDetectThresh', 5);
            p.addOptional('InitExtractWin', -8 : 19);
            p.addOptional('InitNumPC', 3);
            p.addOptional('InitDropClusterThresh', 0.6);
            p.addOptional('InitOverlapTime', 0.4);
            p.addOptional('InitSortDf', 5);
            p.addOptional('InitSortClusterCost', 0.002);
            p.addOptional('InitSortDriftRate', 400 / 3600 / 1000);
            p.addOptional('InitSortTolerance', 0.0005);
            p.addOptional('InitSortCovRidge', 1.5);
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
            elseif ~self.Debug
                delete([self.TempDir '/*'])
            end
            
            self.Fs = 2 * self.NyquistFreq;
        end
        
        
        function delete(self)
            % Class destructor
            
            % remove temp directory unless in debug mode
            if ~self.Debug
                delete(fullfile(self.TempDir, '*'))
                rmdir(self.TempDir)
            end
        end
        
        
        function readData(self, br)
            % Read raw data, downsample and store in local temp file
            
            assert(self.K == getNbChannels(br), ...
                'Dataset and channel layout are incompatible: %d vs. %d channels!', ...
                getNbChannels(br), self.K)
            
            % create memory-mapped Matlab file
            dataFile = fullfile(self.TempDir, 'data.mat');
            self.matfile = matfile(dataFile, 'writable', true);
            if ~exist(dataFile, 'file')
                nBlocksWritten = 0;
                save(dataFile, '-v7.3', 'nBlocksWritten');
            else
                nBlocksWritten = self.matfile.nBlocksWritten;
                if isinf(nBlocksWritten) % file is already complete
                    fprintf('Using existing temp file: %s\n', dataFile)
                    return
                end
            end
            
            % read data, resample, and store to temp file
            Fs = getSamplingRate(br);
            fr = filteredReader(br, filterFactory.createHighpass(self.HighPass(1), self.HighPass(2), Fs));
            blockSize = round(self.BlockSize * Fs);
            pr = packetReader(fr, 1, 'stride', blockSize);
            [p, q] = rat(self.Fs / Fs);
            nBlocks = length(pr);
            lastBlockSize = ceil((length(fr) - (nBlocks - 1) * blockSize) * p / q);
            newBlockSize = ceil(blockSize * p / q);
            self.N = (nBlocks - 1) * newBlockSize + lastBlockSize;
            if ~nBlocksWritten
                h5create(dataFile, '/V', [self.N self.K], 'ChunkSize', [newBlockSize self.K]);
            end
            fprintf('Writing temporary file containing resampled data [%d blocks]\n%s\n', nBlocks, dataFile)
            for i = nBlocksWritten + 1 : nBlocks
                if ~rem(i, 10)
                    fprintf('%d ', i)
                end
                V = toMuV(br, resample(pr(i), p, q));
                start = (i - 1) * newBlockSize;
                self.matfile.V(start + (1 : size(V, 1)), :) = V;
                self.matfile.nBlocksWritten = i;
            end
            self.matfile.nBlocksWritten = inf;
            fprintf('done\n')
        end
        
        
        function X = initialize(self)
            % Initialize model
            
            % load subset of the data
            nskip = ceil(self.N / self.MaxSamples);
            if nskip == 1
                V = self.matfile.V; % load full dataset
            else
                blockSize = self.BlockSize * self.Fs;
                nBlocks = fix(self.N / blockSize);
                subBlockSize = round(blockSize / nskip);
                V = zeros(nBlocks * subBlockSize, size(self.matfile, 'V', 2));
                for i = 1 : nBlocks
                    idxFile = blockSize * (i - 1) + (1 : subBlockSize);
                    idxV = subBlockSize * (i - 1) + (1 : subBlockSize);
                    V(idxV, :) = self.matfile.V(idxFile, :);
                end
            end
            
            % Create channel groups
            channels = self.layout.channelOrder(self.InitChannelOrder);
            num = self.InitNumChannels;
            idx = bsxfun(@plus, 1 : num, (0 : numel(channels) - num)');
            groups = channels(idx);
            nGroups = size(groups, 1);
            
            % Spike sorter
            %   dt needs to be adjusted since we're skipping a fraction of the data
            %   drift rate is per ms, so it needs to be adjusted as well
            m = MoKsm('DTmu', self.BlockSize / nskip * 1000, ...
                'DriftRate', self.InitSortDriftRate * nskip, ...
                'ClusterCost', self.InitSortClusterCost, ...
                'Df', self.InitSortDf, ...
                'Tolerance', self.InitSortTolerance, ...
                'CovRidge', self.InitSortCovRidge);
            
            % detect and sort spikes in groups
            models(nGroups) = m;
            for i = 1 : nGroups
                Vi = V(:, groups(i, :));
                [t, w] = detectSpikes(Vi, self.Fs, self.InitDetectThresh, self.InitExtractWin);
                b = extractFeatures(w, self.InitNumPC);
                models(i) = m.fit(b, t);
            end
            
            % remove duplicate clusters that were created above because the
            % channel groups overlap
            X = self.removeDuplicateClusters(models, self.N);
        end
        
        
        function N = get.N(self)
            if isempty(self.N)
                self.N = size(self.matfile, 'V', 1);
            end
            N = self.N;
        end
        
        
        function m = get.matfile(self)
            if isempty(self.matfile)
                error('Temporary data file not initialized. Run self.readData() first!')
            end
            m = self.matfile;
        end
        
    end
    
    
    methods (Access = private)
        
        function X = removeDuplicateClusters(self, models, N)
            % Remove duplicate clusters.
            %   X = self.keepMaxClusters(models, N) keeps only those
            %   clusters that have their largest waveform on the center
            %   channel. The models are assumed to be fitted to groups of K
            %   channels, with K-1 channels overlap between adjacent
            %   models. Duplicate spikes are removed, keeping the spike
            %   from the cluster with the larger waveform.
            
            % find all clusters having maximum energy on the center channel
            K = self.InitNumChannels;
            center = (K + 1) / 2;
            q = self.InitNumPC;
            nModels = numel(models);
            spikes = {};
            clusters = {};
            mag = [];
            for i = 1 : nModels
                model = models(i);
                a = model.cluster();
                [~, Ndt, M] = size(model.mu);
                for j = 1 : M;
                    nrm = sqrt(sum(sum(reshape(model.mu(:, :, j), [q K Ndt]), 1) .^ 2, 3));
                    [m, idx] = max(nrm);
                    if idx == center || (i == 1 && idx < center) || (i == nModels && idx > center)
                        spikes{end + 1} = model.s(a == j); %#ok<AGROW>
                        clusters{end + 1} = repmat(numel(spikes), size(spikes{end})); %#ok<AGROW>
                        mag(end + 1) = m; %#ok<AGROW>
                    end
                end
            end
            M = numel(spikes);
            spikesPerCluster = cellfun(@numel, spikes);
            
            spikes = cat(1, spikes{:});
            clusters = cat(1, clusters{:});
            
            % order spikes in time
            [spikes, order] = sort(spikes);
            clusters = clusters(order);
            
            % remove smaller spikes from overlaps
            totalSpikes = numel(spikes);
            keep = true(totalSpikes, 1);
            prev = 1;
            refrac = self.InitOverlapTime * self.Fs;
            for i = 2 : totalSpikes
                if spikes(i) - spikes(prev) < refrac
                    if mag(clusters(i)) < mag(clusters(prev))
                        keep(i) = false;
                    else
                        keep(prev) = false;
                        prev = i;
                    end
                else
                    prev = i;
                end
            end
            spikes = spikes(keep);
            clusters = clusters(keep);
            
            % remove clusters that lost too many spikes to other clusters
            frac = hist(clusters, 1 : M) ./ spikesPerCluster;
            keep = true(numel(spikes), 1);
            for i = 1 : M
                if frac(i) < self.InitDropClusterThresh
                    keep(clusters == i) = false;
                end
            end
            spikes = spikes(keep);
            clusters = clusters(keep);
            [~, ~, clusters] = unique(clusters);
            
            % create spike matrix
            X = sparse(spikes, clusters, 1, N, max(clusters));
        end
        
    end
    
end
