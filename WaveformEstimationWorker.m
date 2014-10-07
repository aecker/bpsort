% Binary Pursuit -- Waveform estimation worker class
%
% Estimates the waveforms of all neurons for a single channel. This class
% is used as a worker for ParBP. The single-channel data is kept in memory
% to avoid repeated slow i/o operations.

classdef WaveformEstimationWorker
    
    properties %#ok<*PROP>
        bp      % ParBP object containing parameter settings etc.
        V       % raw data
        channel % channel being processed
        iter    % iteration counter
    end
    
    
    methods
        
        function self = WaveformEstimationWorker(bp, V, channel)
            % Constructor for WaveformEstimationWorker class.
            %   we = WaveformEstimationWorker(bp, V, channel) constructs a
            %   WaveformEstimationWorker based on the BP base object, using
            %   raw data V, which contains the voltage trace of the given
            %   channel.
            
            assert(isa(bp, 'BP'), 'First input must be BP object!')
            assert(isvector(V) && isnumeric(V), 'V must be a numeric vector!')
            assert(isscalar(channel) && ismember(channel, 1 : bp.K), 'Channel must be scalar between 1 and %d!', bp.K)
            self.bp = bp;
            self.V = V(:);
            self.channel = channel;
            self.iter = 1;
        end
        
        
        function run(self)
            % Main loop for worker.
            %   self.run() starts the worker. Whenever the spike times are
            %   ready, the worker estimates the waveforms of all neurons
            %   for its assigned channel.
            
            done = false;
            while ~done
                
                fprintf('Iteration %d\n', self.iter)
                bp = self.bp;
                
                % wait for spike times to be ready or termination signal
                fprintf('Waiting for spike times or termination signal...\n')
                spikeFileBase = fullfile(bp.tempDir, sprintf(bp.spikeFile, self.iter - 1));
                while ~exist([spikeFileBase '.done'], 'file') && ~done
                    done = exist(fullfile(bp.tempDir, bp.completionFile), 'file');
                    pause(0.1)
                end
                
                if ~done
                    % load spike times from file
                    spikeFiles = dir([spikeFileBase '.*.mat']);
                    spikeFiles = {spikeFiles.name};
                    X = loadSpikes(spikeFiles);
                    
                    % estimate waveforms
                    U = bp.estimateWaveforms(self.V, X);
                    
                    % write waveforms to file
                    self.iter = self.iter + 1;
                    waveformFileTmp = fullfile(bp.tempDir, sprintf([bp.waveformFile '.%d.saving'], self.iter, self.channel));
                    waveformFile = fullfile(bp.tempDir, sprintf([bp.waveformFile '.%d.mat'], self.iter, self.channel));
                    fastsave(waveformFileTmp, U)
                    movefile(waveformFileTmp, waveformFile);
                end
            end
        end
        
    end
    
end


function X = loadSpikes(files)
    nChunks = numel(files);
    for k = nChunks : -1 : 1
        d(k) = load(files{k});
    end
    X = sparse(cat(1, d.i), cat(1, d.j), cat(1, d.a) + 1i * cat(1, d.r), sum(cat(1, data.N)), d(1).M);
end
