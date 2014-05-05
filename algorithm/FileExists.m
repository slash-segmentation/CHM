% Workaround for a bug in MATLAB when in that exists(x, 'file') and exists(x, 'dir') give incorrect and inconsistent results
function b = FileExists(x, folder);
if nargin < 1 || nargin > 2; error('FileExists must have 1 or 2 input arguments'); end
if nargin < 2; folder = false; end

[d,f,e] = fileparts(x);
if strcmp(d, ''); x = fullfile('.', [f e]); end;
if folder; b = exist(x, 'dir')==7;
else; b = exist(x, 'file')==2;
end
