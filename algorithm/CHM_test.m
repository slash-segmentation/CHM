function CHM_test(input_files,outputpath,savingpath)
if nargin < 2 || nargin > 3; error('CHM_test must have 2 to 3 input arguments'); end
if nargin == 2; savingpath = fullfile('.', 'temp'); end

param = load(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage');

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

opened_pool = 0;
try; if ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

files_te = GetFiles(input_files);
if exist(outputpath,'file')~=7; mkdir(outputpath); end

% Test CHM
parfor i = 1:length(files_te)
    [~,filename,ext] = fileparts(files_te{i});
    img = imread(files_te{i});
    clabels = testCHM(img,savingpath,param); % uses Nfeatcontext, NLevel, and NStage from param
    %parsave(fullfile(outputpath, filename), clabels);
    imwrite(uint8(clabels*255), fullfile(outputpath, [filename ext]));
end

if opened_pool; matlabpool close; end

function parsave(path, clabels)
save(path,'clabels','-v7.3');
