% Spike sorting using binary pursuit
% Parallel implementation based on BP class
%
% AE 2014-10-07

classdef ParBP < BP
    properties %#ok<*PROP>
        chunkSize   % chunk size (# blocks of length dt sec [see below])
        tempDir     % temporary directory containing files for communication with workers
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
            p.addOptional('chunkSize', 10);
            p.addOptional('tempDir', tempdir());
            p.parse(varargin{:});
            args = [fieldnames(p.Unmatched), struct2cell(p.Unmatched)]';
            self = self@BP(layout, args{:});
            self.chunkSize = p.Results.chunkSize;
            self.tempDir = p.Results.tempDir;
        end
        
    end
    
end
