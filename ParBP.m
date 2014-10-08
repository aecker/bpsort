% Spike sorting using binary pursuit
% Parallel implementation based on BP class
%
% AE 2014-10-07

classdef ParBP < BP
    properties %#ok<*PROP>
        blocksPerChunk  % chunk size (# blocks of length dt sec [see below])
        tempDir         % temporary directory containing files for communication with workers
    end
    
    properties (SetAccess = private)
        samplesPerChunk % number of samples per chunk
    end
    
    properties (Constant)
        spikeFile = 'spikes.%d'         % file containing spikes
        waveformFile = 'waveforms.%d'   % file containing waveforms
        priorFile = 'priors.%d'         % file containing priors
        completionFile = 'done'         % file indicating completion
    end
    
    methods
        
        function self = ParBP(layout, varargin)
            % ParBP constructor
            %   bp = ParBP(layout) constructs a ParBP object using the
            %   specified channel layout.
            %
            %   bp = ParBP(layout, 'param1', value1, 'param2', value2, ...)
            %   can be used to set optional parameters during construction.
            %   For details, see public properties of ParBP and BP classes.
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('blocksPerChunk', 10);
            p.addOptional('tempDir', fullfile(tempdir(), datestr(now(), 'BP_yyyymmdd_HHMMSS')));
            p.parse(varargin{:});
            args = [fieldnames(p.Unmatched), struct2cell(p.Unmatched)]';
            self = self@BP(layout, args{:});
            self.blocksPerChunk = p.Results.blocksPerChunk;
            self.tempDir = p.Results.tempDir;
            if ~exist(self.tempDir, 'file')
                mkdir(self.tempDir)
            else
                delete([self.tempDir '/*'])
            end
            self.samplesPerChunk = self.dt * self.Fs * self.blocksPerChunk;
        end
        
        
        function [X, U] = fit(self, chunks, iter)
            % Fit model.
            
            % initialization and whitening
            % TODO
            
            if nargin < 2
                iter = 3;
            end
            for i = 1 : iter
                
                % wait until waveforms are ready
                done = false;
                while ~done
                    pause(1)
                    waveformFileBase = fullfile(self.tempDir, sprintf(self.waveformFile, iter));
                    if numel(dir([waveformFileBase '.*.mat'])) == self.K
                        done = true;
                        fclose(fopen([waveformFileBase '.done'], 'w'));
                    end
                end
                
                % delete old temp files containing spike times
                delete(fullfile(self.tempDir, sprintf([self.spikeFile '.*'], iter - 1)));
                                
                % wait until spike times are ready
                done = false;
                while ~done
                    pause(1)
                    spikeFileBase = fullfile(self.tempDir, sprintf(self.spikeFile, iter));
                    if numel(dir([spikeFileBase '.*.mat'])) == chunks
                        done = true;
                        fclose(fopen([spikeFileBase '.done'], 'w'));
                    end
                end
                
                % delete old temp files containing waveforms and priors
                delete(fullfile(self.tempDir, sprintf([self.priorFile '.*'], iter - 1)));
                delete(fullfile(self.tempDir, sprintf([self.waveformFile '.*'], iter)));
                
            end
            
        end
        
        
        function delete(self)
            % Class destructor
            
            % remove temp directory
            delete(fullfile(self.tempDir, '*'))
            rmdir(self.tempDir)
        end
        
    end
    
end
