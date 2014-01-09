function CHM_test(input_files,outputpath,savingpath);
if nargin < 2 || nargin > 3; error('CHM_test must have 2 to 3 input arguments'); end
if nargin == 2; savingpath = fullfile('.', 'temp'); end

param = load(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage');

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

files_te  = GetInputFiles(input_files);
files_out = GetOutputFiles(outputpath, files_te);

opened_pool = 0;
try; if length(files_te) > 1 && usejava('jvm') && ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

% Test CHM
parfor i = 1:length(files_te)
    img = imread(files_te{i});
    clabels = testCHM(img,savingpath,param); % uses Nfeatcontext, Nlevel, and Nstage from param
    imwrite(uint8(clabels*255), files_out{i});
end

if opened_pool; matlabpool close; end
