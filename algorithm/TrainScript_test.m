function TrainScript(testpath,savingpath,Nstage,Nlevel);

if ~strcmp(testpath(end),'/')
    testpath = [testpath '/'];
end

if ~strcmp(savingpath(end),'/')
    savingpath = [savingpath '/'];
end

% Test CHM
addpath(genpath('FilterStuff'));
load([savingpath 'param.mat'],'param')
param.Nstage = Nstage;



fileste = dir([testpath '*.png']); 
str = [savingpath 'output_testImages/'];
mkdir(str);
outputs{length(fileste)} = [];
parfor i = 1:length(fileste)
    tic;
    img = imread([testpath fileste(i).name]);
    clabels = testCHM(img,savingpath,param);
    outputs{i} = clabels;
    fprintf('%f\n',toc);
end

fprintf('\n');
for i = 1:length(fileste)
    tic;
    clabels = outputs{i};
    save([str 'slice' num2str(i)],'clabels','-v7.3');
    imwrite(clabels,[str fileste(i).name]);
    fprintf('%f\n', toc);
end


exit force;
