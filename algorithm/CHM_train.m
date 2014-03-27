function CHM_train(trainpath, labelpath, savingpath, Nstage, Nlevel, restart)
% CHM_train   CHM Image Training Phase Script
%   CHM_train(trainpath, labelpath, [savingpath='./temp'], [Nstage=2], [Nstage=4], [restart=0])
%
%   trainpath is a set of data files to train on, see below for details.
%   labelpath is a set of label files to train on, see below for details.
%   savingpath is the folder to save the trained model to (along with temporary files)
%       Only need to keep MODEL_level#_stage#.mat and param.mat files in that folder.
%   Nstage is the number of stages to train with, must be >=2
%   Nlevel is the number of levels to train with, must be >=1
%   restart is if the model should be able to restart a previous model
%
% traingpath and labelpath are comma-seperated lists of the following:
%   path to a folder            - all PNGs in that folder
%   path to a file              - only that file 
%   path with numerical pattern - get all files matching the pattern
%       pattern must have #s in it and end with a semicolon and number range
%       the #s are replaced by the values at the end with leading zeros
%       example: in/####.png;5-15 would do in/0005.png through in/0015.png
%   path with wildcard pattern  - get all files matching the pattern
%       pattern has * in it which means any number of any characters
%       example: in/*.tif does all TIFF images in that directory

if nargin < 2 || nargin > 6; error('CHM_train must have 2 to 6 input arguments'); end
if nargin < 3; savingpath = fullfile('.', 'temp'); end
if nargin < 4; Nstage = 2; end
if nargin < 5; Nlevel = 4; end
if nargin < 6; restart = 0; end

if ~ismcc && ~isdeployed
    % Add path to functions required for feature extraction (already included in compiled version)
    [my_path, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(my_path, 'FilterStuff')));
else
    % Parse non-string arguments
    Nstage = ParseArgument(Nstage);
    Nlevel = ParseArgument(Nlevel);
    restart = ParseArgument(restart);
end

files_tr = GetInputFiles(trainpath);
files_la = GetInputFiles(labelpath);
if numel(files_tr) < 1 || numel(files_tr) ~= numel(files_la); error('You must provide at least 1 image set and equal numbers of training and label images'); end;
if exist(savingpath,'file')~=7; mkdir(savingpath); end

if restart && exist(fullfile(savingpath,'param.mat'),'file') == 2
    param = load(fullfile(savingpath, 'param.mat'), 'Nfeatcontext', 'Nlevel', 'Nstage', 'TrainingSize');
    if GetImageSize(files_tr{1}) ~= param.TrainingSize; error('Cannot restart since training data is a different size'); end;
    % Remove all stages/levels that would be invalid if Nlevel/Nstage change
    files_m = dir(fullfile(savingpath,'MODEL_level*_stage*.mat'));
    for i = 1:length(files_m)
        % TODO too aggressive
        x = sscanf(files_m(i).name, 'MODEL_level%d_stage%d.mat');
        l = x(1); s = x(2);
        if s > Nstage || l > Nlevel || (Nlevel ~= param.Nlevel && s > 1)
            delete(fullfile(savingpath,files_m(i).name));
            [~,~,~] = rmdir(fullfile(savingpath,['output_level' num2str(l) '_stage' num2str(s)]),'s');
        end
    end
    done = 0;
    for Nstage_start = 1:Nstage
        for Nlevel_start = 0:Nlevel
            base_name = ['level' num2str(Nlevel_start) '_stage' num2str(Nstage_start)];
            model_file = fullfile(savingpath, ['MODEL_' base_name '.mat']);
            output_dir = fullfile(savingpath, ['output_' base_name]);
            if exist(model_file,'file') ~= 2 || exist(output_dir,'file') ~= 7; done = 1; break; end;
            for i = 1:length(files_tr)
                [~,filename,~] = fileparts(files_tr{i});
                if exist(fullfile(output_dir,[filename '.mat']),'file') ~= 2; done = 1; break; end;
            end
            if done; break; end;
        end
        if done; break; end;
    end
    fprintf('Restarting from stage %d level %d...\n',Nstage_start,Nlevel_start);
else
    Nstage_start = 1;
    Nlevel_start = 0;
end

% Getting training image size and histogram
im = imread(files_tr{1});
TrainingSize = size(im);
if numel(TrainingSize) ~= 2; error('You must use grayscale images'); end;
[hgram,~] = imhist(im{1});
for i = 2:length(files_tr);
    im = imread(files_tr{i});
    if TrainingSize ~= size(im); error('All training images must have the same dimensions'); end;
    [hg,~] = imhist(im);
    hgram = hgram + hg;
end;

% Only for preallocation purpose
%im = imread(files_tr{1});
Nfeat = size(Filterbank(im),1);
Nfeatcontext = size(ConstructNeighborhoodsS(im),1); % TODO: make shortcut function for getting size?
PixN = zeros(Nlevel+1,1);
% Original: for l = 0:Nlevel; PixN(l+1) = numel(MyDownSample(im,l)); end;
% Faster (2.7x) but assumes MyDownSample(im,n) == MyDownSample(MyDownSample(im,1),n-1) and MyDownSample(im,0) == im
PixN(1) = numel(im);
for l = 1:Nlevel; im = MyDownSample(im,1); PixN(l+1) = numel(im); end
% Even faster (5.1x of original) but assumes that MyDownSample always halves image size (rounded up) for each level
%sz = GetImageSize(files_tr{i});
%PixN(1) = PixN(1) + sz(1)*sz(2);
%for l = 1:Nlevel; sz = floor((sz + 1) / 2); PixN(l+1) = sz(1)*sz(2); end
PixN = PixN*length(files_tr);

save(fullfile(savingpath, 'param.mat'), 'Nfeatcontext', 'Nlevel', 'Nstage', 'TrainingSize', 'hgram', '-v7.3');

param.Nfeat = Nfeat;
param.Nfeatcontext = Nfeatcontext;
param.Nlevel = Nlevel;

opened_pool = 0;
try; if usejava('jvm') && ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

% Train the CHM
for s = Nstage_start:Nstage
    for l = Nlevel_start:Nlevel
        model = trainCHM(files_tr,files_la,savingpath,s,l,PixN(l+1),param); % uses Nfeat, Nfeatcontext, and Nlevel from param
        if s==Nstage, break; end
    end
    Nlevel_start = 0;
end

if opened_pool; matlabpool close; end
