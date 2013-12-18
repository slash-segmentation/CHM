function TrainScript_train(trainpath,labelpath,savingpath,Nstage,Nlevel);

  try 
    if ~strcmp(trainpath(end),'/') 
        trainpath = [trainpath '/'];
    end

    if ~strcmp(labelpath(end),'/')
        labelpath = [labelpath '/'];
    end

    if ~strcmp(savingpath(end),'/')
        savingpath = [savingpath '/'];
    end

    addpath(genpath('FilterStuff')); %add path to functions required for feature extraction.
    mkdir(savingpath);

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

    save([savingpath 'param.mat'],'param');

  catch err
        fprintf('Caught Fatal Exception: %s\n',err.getReport());
        exit(1);
  end

  exit(0);

end
