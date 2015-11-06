classdef BPSorter < BP
    
    properties %#ok<*PROP,*CPROP>
        Debug               % debug mode (true|false)
        TempDir             % temporary folder
        BlockSize           % size of blocks with constant waveform (sec)
        ArtifactBlockSize   % block size used for detecting noise artifacts (sec)
        ArtifactThresh      % threshold for artifact detection (SD of noise in muV)
        MaxSamples          % max number of samples to use
        HighPass            % highpass cutoff [stop, pass] (Hz)
        SubsetIter          % number of iterations on subset after split & merge
        FullIter            % number of full iterations using all data
        
        % properties used for initialization only
        InitChannelOrder    % channel ordering (x|y|xy|yx)
        InitChannelGroupSize     % number of channels to group
        InitChannelGroupStride   % stride for channel grouping
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
        matfile
        N
    end
    
    
    methods
        
        function self = BPSorter(layout, varargin)
            % Constructor for BPSorter class
            
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('Debug', false);
            p.addOptional('TempDir', fullfile(tempdir(), datestr(now(), 'BP_yyyymmdd_HHMMSS')));
            p.addOptional('BlockSize', 60);
            p.addOptional('ArtifactBlockSize', 0.25)
            p.addOptional('ArtifactThresh', 25)
            p.addOptional('MaxSamples', 2e7);
            p.addOptional('HighPass', [400 600]);
            p.addOptional('SubsetIter', 1);
            p.addOptional('FullIter', 1);
            p.addOptional('Fs', 12000);
            p.addOptional('InitChannelOrder', 'y');
            p.addOptional('InitChannelGroupSize', 8);
            p.addOptional('InitChannelGroupStride', 4);
            p.addOptional('InitDetectThresh', 5);
            p.addOptional('InitExtractWin', -8 : 19);
            p.addOptional('InitNumPC', 3);
            p.addOptional('InitDropClusterThresh', 0.6);
            p.addOptional('InitOverlapTime', 0.4);
            p.addOptional('InitSortDf', 5);
            p.addOptional('InitSortClusterCost', 0.002);
            p.addOptional('InitSortDriftRate', 10 / 3600 / 1000);
            p.addOptional('InitSortTolerance', 0.0005);
            p.addOptional('InitSortCovRidge', 1.5);
            p.parse(varargin{:});
            
            assert(~isfield(p.Unmatched, 'dt'), 'Cannot set parameter dt. Use BlockSize instead!')
            self = self@BP(layout, p.Unmatched);
            
            par = fieldnames(p.Results);
            for i = 1 : numel(par)
                self.(par{i}) = p.Results.(par{i});
            end
            assert(rem(self.BlockSize / self.ArtifactBlockSize + 1e-5, 1) < 2e-5, ...
                'BlockSize must be multiple of ArtifactBlockSize!')

            if ~exist(self.TempDir, 'file')
                mkdir(self.TempDir)
            elseif ~self.Debug
                delete([self.TempDir '/*'])
            end
        end
        
        
        function delete(self)
            % Class destructor
            
            % remove temp directory unless in debug mode
            if ~self.Debug
                delete(fullfile(self.TempDir, '*'))
                rmdir(self.TempDir)
            end
        end
        
        
        function set.BlockSize(self, b)
            self.BlockSize = b;
            self.dt = b;
        end
        
        
        function [X, Uw, priors, temporal, spatial] = fit(self, V, X)
            % Fit model.
            
            % read training data if necessary
            if ~exist('V', 'var') || isempty(V)
                self.log(false, 'Loading training data...\n')
                [V, subBlockSize] = self.loadTrainingData();
            end
            
            % initialize on subset of the data using traditional spike
            % detection + sorting algorithm
            if ~exist(X, 'var') || isempty(X)
                self.log(false, 'Initializing model using Mixture of Kalman filter model...\n')
                X = self.initialize(V);
            end
            
            % fit BP model on subset of the data
            self.log('Starting to fit BP model on subset of the data\n\n')
            t = now;
            
            % whiten data
            driftVar = self.BlockSize * self.driftRate;
            v = var(V);
            U = self.estimateWaveforms(V, X, subBlockSize, driftVar);
            [V, temporal, spatial] = self.whitenData(V, self.residuals(V, X, U));
            driftVarWhitened = driftVar / mean(v ./ var(V));
            
            split = true;
            doneSplitMerge = false;
            priors = sum(X > 0, 1) / size(X, 1);
            i = 0;
            M = 0;
            a = 1;
            while i <= self.SubsetIter || ~doneSplitMerge
                
                % estimate waveforms
                Uw = self.estimateWaveforms(V, X, subBlockSize, driftVarWhitened);
                
                % merge templates that are too similar
                if ~doneSplitMerge
                    [Uw, priors, merged] = self.mergeTemplates(Uw, priors);
                end
                
                % stop merging when number of templates does not increase
                % compared to previous iteration
                if numel(priors) <= M || (~split && ~merged)
                    doneSplitMerge = true;
                end
                
                % prune waveforms and estimate spikes
                Uwp = self.pruneWaveforms(Uw);
                [X, priors] = self.estimateSpikes(V, Uwp, priors);
                
                % remove non-spiking templates
                if any(~priors)
                    X = X(:, priors > 0);
                end
                priors = priors(priors > 0);
                M = numel(priors);
                
                % save results from iteration
                if self.Debug
                    save(fullfile(self.TempDir, sprintf('iter%02d', a)), 'X', 'Uw', 'Uwp', 'self', 'priors', 'doneSplitMerge', 'i');
                    a = a + 1;
                end
                
                % split templates with bimodal amplitude distribution
                if ~doneSplitMerge
                    [X, priors, split] = self.splitTemplates(X, priors);
                else
                    i = i + 1;
                end
                
                self.log('\n')
            end
            
            % Order templates spatially
            [Uw, ~, priors] = self.orderTemplates(Uwp, X, priors, 'yx');
            
            % final run in chunks over entire dataset
            [X, Uw] = self.estimateByBlock(Uw, priors, temporal, spatial, driftVarWhitened);
            priors = sum(X > 0, 1) / size(X, 1);
            
            self.log('\n--\nDone fitting model [%.0fs]\n\n', (now - t) * 24 * 60 * 60)
        end
        
        
        function createTempDataFile(self, br)
            % Read raw data, downsample and store in local temp file
            
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
            
            assert(self.K == getNbChannels(br), ...
                'Dataset and channel layout are incompatible: %d vs. %d channels!', ...
                getNbChannels(br), self.K)
            
            % read data, resample, and store to temp file
            raw.Fs = getSamplingRate(br);
            fr = filteredReader(br, filterFactory.createHighpass(self.HighPass(1), self.HighPass(2), raw.Fs));
            raw.blockSize = round(self.BlockSize * raw.Fs);
            raw.artifactBlockSize = round(self.ArtifactBlockSize * raw.Fs);
            nArtifactBlocks = length(fr) / raw.artifactBlockSize;
            nArtifactBlocksPerDataBlock = round(self.BlockSize / self.ArtifactBlockSize);
            raw.N = fix(nArtifactBlocks) * raw.artifactBlockSize;
            nBlocks = ceil(raw.N / raw.blockSize);
            pr = packetReader(fr, 1, 'stride', raw.blockSize);
            [p, q] = rat(self.Fs / raw.Fs);
            new.lastBlockSize = ceil((raw.N - (nBlocks - 1) * raw.blockSize) * p / q);
            new.blockSize = ceil(raw.blockSize * p / q);
            new.artifactBlockSize = round(self.ArtifactBlockSize * self.Fs);
            self.N = (nBlocks - 1) * new.blockSize + new.lastBlockSize;
            if ~nBlocksWritten
                h5create(dataFile, '/V', [self.N self.K], 'ChunkSize', [new.blockSize self.K]);
                h5create(dataFile, '/artifact', [nArtifactBlocks 1], 'DataType', 'uint8');
            end
            fprintf('Writing temporary file containing resampled data [%d blocks]\n%s\n', nBlocks, dataFile)
            for i = nBlocksWritten + 1 : nBlocks
                if ~rem(i, 10)
                    fprintf('%d ', i)
                end
                V = toMuV(br, resample(pr(i), p, q));
                V = bsxfun(@minus, V, mean(V, 2)); % subtract common reference
                if i == nBlocks
                    V = V(1 : new.lastBlockSize, :); % crop to multiple of artifact block size
                end
                
                % detect noise artifacts
                V = reshape(V, [new.artifactBlockSize, size(V, 1) / new.artifactBlockSize, self.K]);
                artifact = any(median(abs(V), 1) / 0.6745 > self.ArtifactThresh, 3);
                artifact = conv(double(artifact), ones(1, 3), 'same') > 0;
                sa = (i - 1) * nArtifactBlocksPerDataBlock;
                artifact(1) = artifact(1) || (i > 1 && self.matfile.artifact(sa, 1));
                V(:, artifact, :) = 0;
                V = reshape(V, [], self.K);
                
                % write to disk
                sb = (i - 1) * new.blockSize;
                self.matfile.V(sb + (1 : size(V, 1)), :) = V;
                if i > 1 && artifact(1)
                    self.matfile.V(sb + (-new.artifactBlockSize : 0), :) = 0;
                    self.matfile.artifact(sa, 1) = true;
                end
                self.matfile.artifact(sa + (1 : numel(artifact)), 1) = artifact(:);
                self.matfile.nBlocksWritten = i;
            end
            self.matfile.nBlocksWritten = inf;
            fprintf('done\n')
        end
        
        
        function X = initialize(self, V)
            % Initialize model using Mixture of Kalman filter clustering
            
            % load initialization files
            modelFile = fullfile(self.TempDir, ...
                sprintf('models%dstride%d.mat', self.InitChannelGroupSize, self.InitChannelGroupStride));
            if exist(modelFile, 'file')
                fprintf('  Using existing initialization.\n')
                load(modelFile, 'models')
            else
            
                % Create channel groups
                channels = self.layout.channelOrder(self.InitChannelOrder);
                num = self.InitChannelGroupSize;
                stride = self.InitChannelGroupStride;
                idx = bsxfun(@plus, 1 : num, (0 : stride : numel(channels) - num)');
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
                tt = struct('w', {cell(1, num)}, 't', cell(1, nGroups));
                parfor i = 1 : nGroups
                    Vi = V(:, groups(i, :)); %#ok<*PFBNS>
                    [t, w] = detectSpikes(Vi, self.Fs, self.InitDetectThresh, self.InitExtractWin);
                    b = extractFeatures(w, self.InitNumPC);
                    models(i) = m.fit(b, t);
                    for j = 1 : num
                        tt(i).w{j} = w(:, :, j); %#ok
                    end
                    tt(i).t = t;
                end
                save(modelFile, 'models', 'tt')
            end
            
            % remove duplicate clusters that were created above because the
            % channel groups overlap
            X = self.removeDuplicateClusters(models, size(V, 1));
        end
        
        
        function [V, subBlockSize] = loadTrainingData(self)
            nskip = ceil(self.N / self.MaxSamples);
            if nskip == 1
                subBlockSize = self.BlockSize * self.Fs;
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
        end
        
        
        function N = get.N(self)
            if isempty(self.N)
                self.N = size(self.matfile, 'V', 1);
            end
            N = self.N;
        end
        
        
        function m = get.matfile(self)
            if isempty(self.matfile)
                error('Temporary data file not initialized. Run self.createTempDataFile() first!')
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
            K = self.InitChannelGroupSize;
            S = self.InitChannelGroupStride;
            center = ceil((K - S) / 2) + (1 : S);
            q = self.InitNumPC;
            nModels = numel(models);
            spikes = {};
            clusters = {};
            mag = [];
            for i = 1 : nModels
                model = models(i);
                a = model.cluster();
                [~, Ndt, M] = size(model.mu);
                for j = 1 : M
                    nrm = sqrt(sum(sum(reshape(model.mu(:, :, j), [q K Ndt]), 1) .^ 2, 3));
                    [m, idx] = max(nrm);
                    if ismember(idx, center) || (i == 1 && idx < center(1)) || (i == nModels && idx > center(end))
                        spikes{end + 1} = round(model.t(a == j) * self.Fs / 1000); %#ok<AGROW>
                        clusters{end + 1} = repmat(numel(spikes), size(spikes{end})); %#ok<AGROW>
                        mag(end + 1) = m; %#ok<AGROW>
                    end
                end
            end
            M = numel(spikes);
            spikesPerCluster = cellfun(@numel, spikes);
            
            spikes = [spikes{:}];
            clusters = [clusters{:}];
            
            % order spikes in time
            [spikes, order] = sort(spikes);
            clusters = clusters(order);
            
            % remove smaller spikes from overlaps
            totalSpikes = numel(spikes);
            keep = true(totalSpikes, 1);
            prev = 1;
            refrac = self.InitOverlapTime * self.Fs / 1000;
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
        
        
        function [X, Uw] = estimateByBlock(self, Uw, priors, temporal, spatial, drift)
            % Estimate model on full dataset working blockwise
            
            % determine active channels for each neuron
            active = permute(max(sum(abs(Uw), 1), [], 4) > 1e-6, [2 3 1]);
            
            blockSize = self.BlockSize * self.Fs;
            nBlocks = ceil(self.N / blockSize);
            if nBlocks > size(Uw, 4) % add waveforms for last (incomplete) block
                Uw(:, :, :, nBlocks) = Uw(:, :, :, end);
            end
            M = numel(priors);
            K = self.K;
            
            X = cell(1, nBlocks);
            for f = 0 : self.FullIter
                
                Uf = cell(K, nBlocks);
                P = cell(K, nBlocks);
                for t = 1 : nBlocks
                    
                    % read data
                    start = (t - 1) * blockSize;
                    idx = start + (1 : min(self.N - start, blockSize));
                    V = self.matfile.V(idx, :);
                    
                    % whiten data
                    for k = 1 : self.K
                        V(:, k) = conv(V(:, k), temporal(:, k), 'same');
                    end
                    V = V * spatial;
                    
                    % estimate spikes
                    X{t} = self.estimateSpikes(V, Uw(:, :, :, t), priors);
                    
                    % estimate waveforms (forward pass)
                    if f < self.FullIter
                        tt = t - 1;
                        [Uf(:, t), P(:, t)] = self.estimateWaveformsFwdPass(V, X{t}, active, Uf(:, tt(tt > 0)), P(:, tt(tt > 0)), drift);
                    end
                end
                
                if f < self.FullIter
                    % Backward pass
                    E = size(Uw, 1);
                    U = zeros(E * M, K, nBlocks);
                    for k = 1 : K
                        a = reshape(repmat(active(k, :), E, 1), [], 1);
                        U(a, k, end) = Uf{k, end};
                        for t = nBlocks - 1 : -1 : 1
                            Ct = P{k, t} / P{k, t + 1};
                            U(a, k, t) = Uf{k, t} + Ct * (Uf{k, t + 1} - Uf{k, t});
                        end
                    end
                    
                    % Re-organize waveforms by cluster
                    Uw = reshape(U, [E M K nBlocks]);
                    Uw = permute(Uw, [1 3 2 4]);
                end
            end
            
            % create spike matrix
            [i, j, x] = cellfun(@find, X, 'uni', false);
            i = arrayfun(@(i, s) i{1} + s, i, (0 : nBlocks - 1) * blockSize, 'uni', false);
            X = sparse(cat(1, i{:}), cat(1, j{:}), cat(1, x{:}), self.N, M);
        end
        
        
        function [U, P] = estimateWaveformsFwdPass(self, V, X, active, U, P, drift)
            
            % Pre-compute convolution matrix: MX * W = conv(X, W)
            [i, j, x] = find(X);
            r = imag(x);
            a = real(x);
            d = 2 * (r > 0) - 1;
            i = [i; i + d];
            i = bsxfun(@plus, i, self.samples);
            valid = find(i > 0 & i <= size(V, 1));
            D = self.D;
            j = bsxfun(@plus, (j - 1) * D, 1 : D);
            j = [j; j];
            x = repmat([a .* (1 - abs(r)); a .* abs(r)], 1, D);
            
            [i, order] = sort(i(valid));
            j = j(valid(order));
            x = x(valid(order));
            
            [Tdt, M] = size(X);
            K = self.K;
            B = self.waveformBasis;
            if isempty(B)
                E = D;
            else
                E = size(B, 2);
            end
            
            % Pre-compute MX' * MX
            MX = sparse(i, j, x, Tdt, D * M);
            BMXp = BP.getBMXprod(MX, B);
            
            % Initialize or forward step?
            if isempty(U)
                
                MXV = MX' * V;
                if isempty(B)
                    BMXV = MXV;
                else
                    BMXV = zeros(E * M, K);
                    for m = 1 : M
                        iD = (m - 1) * D + (1 : D);
                        iE = (m - 1) * E + (1 : E);
                        BMXV(iE, :) = B' * MXV(iD, :);
                    end
                end
                
                % using pinv() instead of \ because MXprod can be rank-
                % deficient if there are no or only few spikes for some neurons
                U1 = pinv(BP.getBMXprod(MX, B)) * BMXV;
                
                % Initialize state covariance
                n = full(sum(MX, 1));
                P = diag(1 ./ (n + ~n));
                if ~isempty(B)
                    P1 = zeros(E * M);
                    for m = 1 : M
                        iD = (m - 1) * D + (1 : D);
                        iE = (m - 1) * E + (1 : E);
                        P1(iE, iE) = B' * P(iD, iD) * B;
                    end
                else
                    P1 = P;
                end
                
                % account for pruning
                U = cell(K, 1);
                P = cell(K, 1);
                for k = 1 : K
                    a = reshape(repmat(active(k, :), E, 1), [], 1);
                    U{k} = U1(a, k);
                    P{k} = P1(a, a);
                end
            else 
                for k = 1 : K
                    
                    Mk = sum(active(k, :));
                    aE = reshape(repmat(active(k, :), E, 1), [], 1);
                    aD = reshape(repmat(active(k, :), D, 1), [], 1);
                    I = eye(E * Mk);
                    Q = I * drift;
                    
                    % Predict
                    Pk = P{k} + Q;
                    Uk = U{k};
                    
                    % Update
                    Kp = Pk * (I - BMXp(aE, aE) / (inv(Pk) + BMXp(aE, aE))); % Kalman gain (K = Kp * MX)
                    KpBMXp = Kp * BMXp(aE, aE);
                    MXV = MX(:, aD)' * V(:, k);
                    if isempty(B)
                        BMXV = MXV;
                    else
                        BMXV = zeros(E * Mk, 1);
                        for m = 1 : Mk
                            iD = (m - 1) * D + (1 : D);
                            iE = (m - 1) * E + (1 : E);
                            BMXV(iE) = B' * MXV(iD);
                        end
                    end
                    U{k} = Uk + Kp * BMXV - KpBMXp * Uk;
                    P{k} = (I - KpBMXp) * Pk;
                end
            end
        end
        
    end
    
end
