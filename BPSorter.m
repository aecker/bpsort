classdef BPSorter < handle
    
    properties %#ok<*PROP,*CPROP>
        Debug               % debug mode (true|false)
        TempDir             % temporary folder
        BlockSize           % size of blocks with constant waveform (sec)
        MaxSamples          % max number of samples to use
        HighPass            % highpass cutoff [stop, pass] (Hz)
        NyquistFreq         % Nyquist frequency (Hz)
        DropClusterThresh   % threshold for dropping clusters
        
        % properties used for initialization only
        InitChannelOrder    % channel ordering (x|y|xy|yx)
        InitNumChannels     % number of channels to group
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
            p.addOptional('Debug', false);
            p.addOptional('TempDir', fullfile(tempdir(), datestr(now(), 'BP_yyyymmdd_HHMMSS')));
            p.addOptional('BlockSize', 60);
            p.addOptional('MaxSamples', 2e7);
            p.addOptional('HighPass', [400 600]);
            p.addOptional('NyquistFreq', 6000);
            p.addOptional('DropClusterThresh', 0.6);
            p.addOptional('InitChannelOrder', 'y');
            p.addOptional('InitNumChannels', 5);
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
            h5create(dataFile, '/V', [self.N self.K], 'ChunkSize', [newBlockSize self.K]);
            fprintf('Creating temporary file containing resampled data [%d blocks]\n%s\n', nBlocks, dataFile)
            for i = nBlocksWritten + 1 : nBlocks
                if ~rem(i, 10)
                    fprintf('%d ', i)
                end
                V = toMuV(br, resample(pr(i), p, q));
                start = (i - 1) * newBlockSize;
                self.matfile.V(start + (1 : size(V, 1)), :) = V;
                self.matfile.nBlocksWritten = i;
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
            num = self.InitNumChannels;
            idx = bsxfun(@plus, 1 : num, (0 : numel(channels) - num)');
            groups = channels(idx);
            nGroups = size(groups, 1);
            
            % detect and sort spikes in groups
            for i = 1 : nGroups
                Vi = V(:, groups(i, :));
                [t, w] = detectSpikes(Vi, self.Fs, self.InitDetectThresh, self.InitExtractWin);
                b = extractFeatures(w, self.InitNumPC);
                
                % dt needs to be adjusted since we're skipping a fraction of the data
                % drift rate is per ms, so it needs to be adjusted as well
                model = MoKsm('DTmu', par.BlockSize / nskip * 1000, 'DriftRate', par.DriftRate * nskip, ...
                    'ClusterCost', par.ClusterCost, 'Df', par.Df, 'Tolerance', par.Tolerance, 'CovRidge', par.CovRidge);
                model = model.fit(b, t);
                
                % NEED TO DO SOMETHING WITH IT!!
            end
        end
        
    end
    
end
