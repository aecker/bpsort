% Binary Pursuit -- Spike time estimation worker class
%
% Estimates the spike times of all neurons for a chunk of data. This class
% is used as a worker for ParBP. The data chunk is kept in memory to avoid
% repeated slow i/o operations.

classdef SpikeTimeEstimationWorker
    
    properties %#ok<*PROP>
        bp      % ParBP object containing parameter settings etc.
        V       % raw data
        chunk   % chunk being processed
        iter    % iteration counter
    end
    
    
    methods
        
        function self = SpikeTimeEstimationWorker(bp, V, chunk)
            % Constructor for SpikeTimeEstimationWorker class.
            %   we = SpikeTimeEstimationWorker(bp, V, chunk) constructs a
            %   SpikeTimeEstimationWorker based on the ParBP object bp,
            %   using raw data V, which contains the voltage trace of the
            %   given chunk.
            
            assert(isa(bp, 'ParBP'), 'First input must be ParBP object!')
            assert(all(size(V) == [bp.Tc, bp.K]), 'V must be a numeric matrix of size %d-by-%d!', bp.Tc, bp.C)
            assert(isscalar(chunk) && ismember(chunk, 1 : bp.C), 'Chunk number must be scalar between 1 and %d!', bp.C)
            self.bp = bp;
            self.V = V;
            self.chunk = chunk;
            self.iter = 1;
        end
        
        
        function run(self)
            % Main loop for worker.
            %   self.run() starts the worker. Whenever the waveform
            %   templates are ready, the worker estimates the spike times
            %   of all neurons in its assigned data chunk.
            
            done = false;
            while ~done
                
                fprintf('Iteration %d\n', self.iter)
                bp = self.bp;
                
                % wait for waveforms to be ready or termination signal
                fprintf('  Waiting for waveforms or termination signal...\n')
                waveformFileBase = fullfile(bp.tempDir, sprintf(bp.spikeFile, self.iter));
                while ~exist([waveformFileBase '.done'], 'file') && ~done
                    done = exist(fullfile(bp.tempDir, bp.completionFile), 'file');
                    pause(0.1)
                end
                
                if done
                    fprintf('\nAlgorithm terminated\n\n')
                else
                    
                    % load priors and waveforms from file
                    fprintf('  Loading waveform templates... '); tic
                    priorFiles = dir([fullfile(bp.tempDir, sprintf(bp.priorFile, self.iter - 1)) '.*.mat']);
                    priorFiles = {priorFiles.name};
                    priors = loadPriors(priorFiles);
                    M = numel(priors);
                    waveformFiles = dir([waveformFileBase '.*.mat']);
                    waveformFiles = {waveformFiles.name};
                    U = self.loadWaveforms(waveformFiles, M);
                    fprintf('%.1 sec\n', toc)
                    
                    % estimate spike times
                    fprintf('  Estimating spike times... '); tic
                    [X, priors] = bp.estimateSpikes(self.V, U, priors);
                    fprintf('%.1 sec\n', toc)
                    
                    % write spike times and priors to file
                    fprintf('  Writing priors and spike times to disc... '); tic
                    priorFile = fullfile(bp.tempDir, sprintf([bp.priorFile '.%d.mat'], self.iter, self.chunk));
                    fastsave(priorFile, priors);
                    [i, j, x] = find(X);
                    a = real(x);
                    r = imag(x);
                    [N, M] = size(X);
                    spikeFileTmp = fullfile(bp.tempDir, sprintf([bp.spikeFile '.%d.saving'], self.iter, self.chunk));
                    spikeFile = fullfile(bp.tempDir, sprintf([bp.spikeFile '.%d.mat'], self.iter, self.chunk));
                    fastsave(spikeFileTmp, i, j, a, r, N, M);
                    movefile(spikeFileTmp, spikeFile);
                    fprintf('%.1 sec\n', toc)
                end
                
                self.iter = self.iter + 1;
            end
        end
        
        
        function U = loadWaveforms(self, files, M)
            bp = self.bp;
            nChannels = numel(files);
            U = zeros(bp.D, bp.K, M, bp.blocksPerChunk);
            for k = nChannels : -1 : 1
                m = matfile(files{k});
                idx = bp.blocksPerChunk * (self.chunk - 1) + (1 : bp.blocksPerChunk);
                U(:, k, :, :) = m.U(:, 1, :, idx);
            end
        end

    end
    
end



function priors = loadPriors(files)
    nChunks = numel(files);
    for k = nChunks : -1 : 1
        d(k) = load(files{k});
    end
    priors = mean(cat(1, d.priors), 1);
end
