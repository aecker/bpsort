% startup script for BP library

p = fileparts(mfilename('fullpath'));
addpath(p)
addpath(fullfile(p, 'util'))
addpath(fullfile(p, 'util', 'init'))
addpath(fullfile(p, 'plot'))
addpath(fullfile(p, 'data'))

p = fileparts(p);
moksm = fullfile(p, 'moksm');
if exist(moksm, 'file')
    addpath(moksm)
else
    warning('Could not find moksm repo, which is required for initialization!')
end
