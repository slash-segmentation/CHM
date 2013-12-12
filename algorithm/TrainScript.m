function TrainScript(trainpath,labelpath,testpath,savingpath);
if ~matlabpool('size')
	matlabpool open 12;
end

if ~strcmp(trainpath(end),'/') 
    trainpath = [trainpath '/'];
end

if ~strcmp(testpath(end),'/')
    testpath = [testpath '/'];
end

if ~strcmp(labelpath(end),'/')
    labelpath = [labelpath '/'];
end

if ~strcmp(savingpath(end),'/')
    savingpath = [savingpath '/'];
end


%trainpath = './trainImages/'; %path to the training Images
%labelpath = './trainLabels/'; %path to the training Labels
%savingpath = './temp/'; %path to save results and temporary files
mkdir(savingpath);
addpath(genpath('FilterStuff')); %add path to functions required for feature extraction.
% parameters
Nstage = 2;
Nlevel = 4;
% Only for preallocation purpose
filestr = dir([trainpath '*.png']);
ntr = length(filestr);
PixN = zeros(Nlevel+1,1);
for l = 0:Nlevel
    for i = 1:ntr
        temp = MyDownSample(imread([trainpath filestr(i).name]),l);
        PixN(l+1) = PixN(l+1) + numel(temp);
    end
end
tempfeat = Filterbank(imread([trainpath filestr(1).name]));
Nfeat = size(tempfeat,1);
tempfeat = ConstructNeighborhoodsS(imread([trainpath filestr(1).name]));
Nfeatcontext = size(tempfeat,1);

param.ntr = ntr;
param.PixN = PixN;
param.Nfeat = Nfeat;
param.Nfeatcontext = Nfeatcontext;
param.Nlevel = Nlevel;


% Train the CHM
for s = 1:Nstage
    param.stage = s;
    for l = 0:Nlevel
        param.level = l;
        param.PixN = PixN(l+1);
        model = trainCHM(trainpath,labelpath,savingpath,param);
        if s==Nstage, break; end
    end
end

% Test CHM
param.Nstage = Nstage;
%testpath = './testImages/'; %path to the test Images
fileste = dir([testpath '*.png']); 
str = [savingpath 'output_testImages/'];
mkdir(str);
outputs{length(fileste)} = [];
parfor i = 1:length(fileste)
    img = imread([testpath fileste(i).name]);
    clabels = testCHM(img,savingpath,param);
    outputs{i} = clabels;
end

for i = 1:length(fileste)
 clabels = outputs{i};
 save([str 'slice' num2str(i)],'clabels','-v7.3');
 imwrite(clabels,[str fileste(i).name]);
end
if matlabpool('size')
	matlabpool close
end
    
