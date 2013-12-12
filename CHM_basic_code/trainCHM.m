function model = trainCHM(trainpath,labelpath,savingpath,param)


ntr = param.ntr;
PixN = param.PixN;
Nfeat = param.Nfeat;
level = param.level;
stage = param.stage;
Nfeatcontext = param.Nfeatcontext;
Nlevel = param.Nlevel;

filestr = dir([trainpath '*.png']);
filesla = dir([labelpath '*.png']);


% Feature Extraction
fprintf('Extracting features ... stage %d level %d \n',stage,level);

if stage==1 || level~=0
    X = zeros(Nfeat + level*Nfeatcontext,PixN);
    DD = zeros(1,PixN);
    COUNTER = 0;
    for i = 1:ntr
        img = imread([trainpath filestr(i).name]);
        img = MyDownSample(img,level);
        fv = Filterbank(img);

        fvcontext = zeros(level*Nfeatcontext,numel(img));
        for j = 0:level-1
            Contextpath = [savingpath 'output_level' num2str(j) '_stage' num2str(stage) '/'];
            temp = load([Contextpath 'slice' num2str(i)]);
            temp = MyDownSample(temp.clabels,level-j);
            fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
        end

        temp = imread([labelpath filesla(i).name]);
        clabels = double(temp>0);
        clabels = Mymaxpooling(clabels,level);
        D = reshape(clabels,1,numel(clabels));

        X(:,COUNTER+1:COUNTER+numel(img)) = [fv;fvcontext];
        DD(1,COUNTER+1:COUNTER+numel(img)) = D;
        COUNTER = COUNTER + numel(img);
    end
else
    X = zeros(Nfeat + (Nlevel+1)*Nfeatcontext,PixN);
    DD = zeros(1,PixN);
    COUNTER = 0;
    for i = 1:ntr
        img = imread([trainpath filestr(i).name]);
        img = MyDownSample(img,level);
        fv = Filterbank(img);

        fvcontext = zeros((Nlevel+1)*Nfeatcontext,numel(img));
        for j = 0:Nlevel
            Contextpath = [savingpath 'output_level' num2str(j) '_stage' num2str(stage-1) '/'];
            temp = load([Contextpath 'slice' num2str(i)]);
            temp = MyUpSample(temp.clabels,j);
            temp = temp(1:size(img,1),1:size(img,2),:);
            fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
        end

        temp = imread([labelpath filesla(i).name]);
        clabels = double(temp>0);
        clabels = Mymaxpooling(clabels,level);
        D = reshape(clabels,1,numel(clabels));

        X(:,COUNTER+1:COUNTER+numel(img)) = [fv;fvcontext];
        DD(1,COUNTER+1:COUNTER+numel(img)) = D;
        COUNTER = COUNTER + numel(img);
    end
end
% Subsampling
if PixN > 6000000 % increase this for real problems
    ind = [];
    maxs = 3000000;
    for i=0:max(DD)
        if histc(DD,i)>maxs
            a = find(DD==i);
            rpm = randperm(numel(a));
            index = a(rpm(1:maxs));
        else
            index = find(DD==i);
        end
        ind = [ind index];
    end
    X = X(:,ind);
    DD = DD(:,ind);
end
% Learning the Classifier
fprintf('start learning LDNN ... stage %d level %d \n',stage,level);
opt.level = level;
opt.stage = stage;
tic;model = LearnAndOrNetMEX( X,DD,opt);Time = toc;
save([savingpath 'MODEL_level' num2str(level) '_stage' num2str(stage)] , 'model','Time', '-v7.3');
% Write the outputs
fprintf('Generating outputs ... stage %d level %d \n',stage,level);
str = [savingpath 'output_level' num2str(level) '_stage' num2str(stage) '/'];
mkdir(str);

trOutput{ntr} = [];
if stage==1 || level~=0
    parfor i = 1:ntr
        img = imread([trainpath filestr(i).name]);
        img = MyDownSample(img,level);
        fv = Filterbank(img);

        fvcontext = zeros(level*Nfeatcontext,numel(img));
        for j = 0:level-1
            Contextpath = [savingpath 'output_level' num2str(j) '_stage' num2str(stage) '/'];
            temp = load([Contextpath 'slice' num2str(i)]);
            temp = MyDownSample(temp.clabels,level-j);
            fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
        end

        X = [fv;fvcontext];
        [~, Y_floats] = EvaluateAndOrNetMX(X,model);
        clabels = reshape(Y_floats,size(img));
        trOutput{i} = clabels;
    end
else
    parfor i = 1:ntr
        img = imread([trainpath filestr(i).name]);
        img = MyDownSample(img,level);
        fv = Filterbank(img);

        fvcontext = zeros((Nlevel+1)*Nfeatcontext,numel(img));
        for j = 0:Nlevel
            Contextpath = [savingpath 'output_level' num2str(j) '_stage' num2str(stage-1) '/'];
            temp = load([Contextpath 'slice' num2str(i)]);
            temp = MyUpSample(temp.clabels,j);
            temp = temp(1:size(img,1),1:size(img,2),:);
            fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
        end

        X = [fv;fvcontext];
        [~, Y_floats] = EvaluateAndOrNetMX(X,model);
        clabels = reshape(Y_floats,size(img));
        trOutput{i} = clabels;
    end
end

for i = 1:ntr
    clabels = trOutput{i};
    save([str 'slice' num2str(i)],'clabels','-v7.3');
end


