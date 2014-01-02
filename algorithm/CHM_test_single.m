function CHM_test_single(input_file,output_file,savingpath)
if nargin < 2 || nargin > 3; error('CHM_test_single must have 2 to 3 input arguments'); end
if nargin == 2; savingpath = fullfile('.', 'temp'); end

param = load(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage');

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

% Test CHM
img = imread(input_file);
clabels = testCHM(img,savingpath,param); % uses Nfeatcontext, NLevel, and NStage from param
%[folder,filename,~] = fileparts(output_file);
%save(fullfile(folder,filename),'clabels','-v7.3');
imwrite(clabels,output_file);
