function CHM_train(trainpath, labelpath, savingpath, Nstage, Nlevel)
if nargin < 2 || nargin > 5; error('CHM_train must have 2 to 5 input arguments'); end
if nargin < 3; savingpath = fullfile('.', 'temp'); end
if nargin < 4; Nstage = 2; end
if nargin < 5; Nlevel = 4; end

if ~ismcc && ~isdeployed
    % Add path to functions required for feature extraction (already included in compiled version)
    [my_path, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(my_path, 'FilterStuff')));
else
    % Parse non-string arguments
    Nstage = ParseArgument(Nstage);
    Nlevel = ParseArgument(Nlevel);
end

files_tr = GetInputFiles(trainpath);
files_la = GetInputFiles(labelpath);
if numel(files_tr) < 1 || numel(files_tr) ~= numel(files_la); error('You must provide at least 1 image set and equal numbers of training and label images'); end;
if exist(savingpath,'file')~=7; mkdir(savingpath); end

% Only for preallocation purpose
PixN = zeros(Nlevel+1,1);
for i = 1:length(files_tr)
    im = imread(files_tr{i});
    if i == 1;
        TrainingSize = size(im);
        if numel(TrainingSize) ~= 2; error('You must use grayscale images'); end;
    elseif TrainingSize ~= size(im); error('All training images must have the same dimensions'); end;
    
    % Original: for l = 0:Nlevel; PixN(l+1) = PixN(l+1) + numel(MyDownSample(im,l)); end;
    
    % Faster (2.7x) but assumes MyDownSample(im,n) == MyDownSample(MyDownSample(im,1),n-1) and MyDownSample(im,0) == im
    PixN(1) = PixN(1) + numel(im);
    for l = 1:Nlevel
        im = MyDownSample(im,1);
        PixN(l+1) = PixN(l+1) + numel(im);
    end
    
    % Even faster (5.1x of original) but assumes that MyDownSample always halves image size (rounded up) for each level
    %[w,h] = size(imread(files_tr{i}));
    %PixN(1) = PixN(1) + w*h;
    %for l = 1:Nlevel
    %    w = floor((w + 1) / 2);
    %    h = floor((h + 1) / 2);
    %    PixN(l+1) = PixN(l+1) + w*h;
    %end
end

im = imread(files_tr{1});
Nfeat = size(Filterbank(im),1);
Nfeatcontext = size(ConstructNeighborhoodsS(im),1); % TODO: make shortcut function for getting size?

save(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage', 'TrainingSize', '-v7.3');

param.Nfeat = Nfeat;
param.Nfeatcontext = Nfeatcontext;
param.Nlevel = Nlevel;

opened_pool = 0;
try; if usejava('jvm') && ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

% Train the CHM
for s = 1:Nstage
    for l = 0:Nlevel
        model = trainCHM(files_tr,files_la,savingpath,s,l,PixN(l+1),param); % uses Nfeat, Nfeatcontext, and Nlevel from param
        if s==Nstage, break; end
    end
end

if opened_pool; matlabpool close; end
