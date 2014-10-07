function fastsave(filename, varargin)
% Uncompressed fast saves of arrays to .mat files
%   fastsave(filename, A, B, ...) saves variables A, B, ... into the given
%   file. Unlike with MATLAB's built-in save command, A, B, ... are actual
%   variables, not strings containing the variable names.
%
% Matlab's 'save' command can be very slow when saving large arrays,
% because by default Matlab attempts to use compression. This function
% provides a much faster alternative for saving arrays. Only saving of
% numeric (non-sparse) arrays is supported.
%
% Based on savefast.m, originally developed by Timothy E. Holy
% http://www.mathworks.com/matlabcentral/fileexchange/39721-save-mat-files-more-quickly

% Ensure inputs are non-sparse numeric arrays
assert(all(cellfun(@(x) isa(x, 'numeric') && ~issparse(x), varargin)), ...
    'Variables must be numeric and non-sparse!');

% Append .mat if necessary
[filepath, filebase, ext] = fileparts(filename);
if isempty(ext)
    filename = fullfile(filepath, [filebase '.mat']);
end

% Save a dummy variable, just to create the file and then delete it again
foo = 0; %#ok<NASGU>
save(filename, '-v7.3', 'foo');
fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
H5L.delete(fid, 'foo', 'H5P_DEFAULT');
H5F.close(fid);

% Save variables
for i = 1 : numel(varargin)
    varname = ['/' inputname(i + 1)];
    h5create(filename, varname, size(varargin{i}), 'DataType', class(varargin{i}));
    h5write(filename, varname, varargin{i});
end
