function TrainScript(trainpath,labelpath,testpath,outputpath,savingpath);
if nargin < 4 || nargin > 5; error('TrainScript must have 4 to 5 input arguments'); end
if nargin == 4; savingpath = fullfile('.', 'temp'); end

opened_pool = 0;
try; if ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

%trainpath  = fullfile('.', 'trainImages'); %path to the training images
%labelpath  = fullfile('.', 'trainLabels'); %path to the training labels
%testpath   = fullfile('.', 'testImages');  %path to the test images
%savingpath = fullfile('.', 'temp');        %path to save temporary files to
%outputpath = fullfile('.', 'output');      %path to save output images to

files_tr = GetFiles(trainpath);
files_la = GetFiles(labelpath);
files_te = GetFiles(testpath);

if exist(savingpath,'file')~=7; mkdir(savingpath); end

%add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));


% parameters
Nstage = 2;
Nlevel = 4;
% Only for preallocation purpose
PixN = zeros(Nlevel+1,1);
for i = 1:length(files_tr)
    im = imread(files_tr{i});
    for l = 0:Nlevel
        temp = MyDownSample(im,l);
        PixN(l+1) = PixN(l+1) + numel(temp);
    end
end
%im = imread(files_tr{1});
tempfeat = Filterbank(im);
Nfeat = size(tempfeat,1);
tempfeat = ConstructNeighborhoodsS(im);
Nfeatcontext = size(tempfeat,1);

save(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage', '-v7.3'); % don't need to save this for this run, but to be able to run CHM_test_single/CHM_test_multiple later we need it

param.Nfeat = Nfeat;
param.Nfeatcontext = Nfeatcontext;
param.Nlevel = Nlevel;
param.Nstage = Nstage;


% Train the CHM
for s = 1:Nstage
    for l = 0:Nlevel
        model = trainCHM(files_tr,files_la,savingpath,s,l,PixN(l+1),param); % uses Nfeat, Nfeatcontext, and NLevel from param
        if s==Nstage, break; end
    end
end

% Test CHM
if exist(outputpath,'file')~=7; mkdir(outputpath); end
parfor i = 1:length(files_te)
    [~,filename,ext] = fileparts(files_te{i});
    img = imread(files_te{i});
    clabels = testCHM(img,savingpath,param); % uses Nfeatcontext, NLevel, and NStage from param
    %parsave(fullfile(outputpath, filename), clabels);
    imwrite(clabels, fullfile(outputpath, [filename ext]));
end

if opened_pool; matlabpool close; end


function parsave(path, clabels)
save(path,'clabels','-v7.3');
