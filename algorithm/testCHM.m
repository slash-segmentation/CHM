function output = testCHM(I,savingpath,param)

Nfeatcontext = param.Nfeatcontext;
Nlevel = param.Nlevel;
Nstage = param.Nstage;
clabels{Nlevel} = [];
for stage = 1:Nstage
    
    for level = 0:Nlevel

        if stage==1 || level~=0
            img = MyDownSample(I,level);
            fv = Filterbank(img);

            fvcontext = zeros(level*Nfeatcontext,numel(img));
            for j = 0:level-1
                temp = clabels{j+1};
                temp = MyDownSample(temp,level-j);
                fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
            end
            X = [fv;fvcontext];
            load(fullfile(savingpath, ['MODEL_level' num2str(level) '_stage' num2str(stage) '.mat']));
            [~, Y_floats] = EvaluateAndOrNetMX(X,model);
            clabels{level+1} = reshape(Y_floats,size(img));

        else
            img = MyDownSample(I,level);
            fv = Filterbank(img);

            fvcontext = zeros((Nlevel+1)*Nfeatcontext,numel(img));
            for j = 0:Nlevel
                temp = clabels{j+1};
                temp = MyUpSample(temp,j);
                temp = temp(1:size(img,1),1:size(img,2),:);
                fvcontext(j*Nfeatcontext+1:(j+1)*Nfeatcontext,:) = ConstructNeighborhoodsS(temp);
            end
            X = [fv;fvcontext];
            load(fullfile(savingpath, ['MODEL_level' num2str(level) '_stage' num2str(stage) '.mat']));
            [~, Y_floats] = EvaluateAndOrNetMX(X,model);
            clabels{level+1} = reshape(Y_floats,size(img));
        end
        
        if stage==Nstage, output = clabels{1}; break; end
    end
end