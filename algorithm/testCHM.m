function output = testCHM(I,savingpath,param)

Nfeatcontext = param.Nfeatcontext;
clabels{param.Nlevel+1} = [];
Nlevel = param.Nlevel;
for stage = 1:param.Nstage
    
    for level = 0:param.Nlevel

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
            load([savingpath 'MODEL_level' num2str(level) '_stage' num2str(stage)]);
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
            load([savingpath 'MODEL_level' num2str(level) '_stage' num2str(stage)]);
            [~, Y_floats] = EvaluateAndOrNetMX(X,model);
            clabels{level+1} = reshape(Y_floats,size(img));
        end
        
        if stage==param.Nstage, output = clabels{1}; break; end
    end
end