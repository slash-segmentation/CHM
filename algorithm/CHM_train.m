function CHM_train(trainpath, labelpath, savingpath)
if nargin < 2 || nargin > 3; error('CHM_train must have 2 to 3 input arguments'); end
if nargin == 2; savingpath = fullfile('.', 'temp'); end

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

files_tr = GetFiles(trainpath);
files_la = GetFiles(labelpath);
if exist(savingpath,'file')~=7; mkdir(savingpath); end

% Parameters
Nstage = 2;
Nlevel = 4;

% Only for preallocation purpose
PixN = zeros(Nlevel+1,1);
tic;
for i = 1:length(files_tr)
    im = imread(files_tr{i});
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
toc

im = imread(files_tr{1});
tic;
Nfeat = Filterbank('count'); % size(Filterbank(im),1);
toc
tic;
Nfeatcontext = size(ConstructNeighborhoodsS(im),1); % TODO: make shortcut function for getting size?
toc

save(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage', '-v7.3');

param.Nfeat = Nfeat;
param.Nfeatcontext = Nfeatcontext;
param.Nlevel = Nlevel;

opened_pool = 0;
try; if ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

% Train the CHM
for s = 1:Nstage
    for l = 0:Nlevel
        model = trainCHM(files_tr,files_la,savingpath,s,l,PixN(l+1),param); % uses Nfeat, Nfeatcontext, and NLevel from param
        if s==Nstage, break; end
    end
end

if opened_pool; matlabpool close; end
