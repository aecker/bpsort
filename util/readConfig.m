function config = readConfig(configFile)
% Read configuration file.
%   config = readConfig(configFile) read the given configuration file. Each
%   row in the configuration file contains the channel number and the x and
%   y coordinate of the electrode on the probe. The returned struct config
%   contains these values in the fields 'channels', 'x' and 'y'.

fid = fopen(configFile);
if fid
    data = fscanf(fid, '%d\t%g\t%g\n');
    config.name = configFile;
    config.channels = data(1 : 3 : end);
    config.x = data(2 : 3 : end);
    config.y = data(3 : 3 : end);
    fclose(fid);
else
    error('Could not open config file %s!', configFile)
end
