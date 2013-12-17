function TrainScript_testBlocks( testpath, savingpath, combineOutFile, Nstage, Nlevel, BlockSize )

    try 

        TrainScript_testBlocksTimer = tic;

        if ~strcmp(testpath(end),'/')
            testpath = [testpath '/'];
        end

        if ~strcmp(savingpath(end),'/')
            savingpath = [savingpath '/'];
        end

        addpath(genpath('FilterStuff'));
        load([savingpath 'param.mat'],'param');
        mkdir([savingpath 'output_testImages']);
        param.Nstage = Nstage;

        N_test = dir([testpath '*.png']);

        for j = 1:numel(N_test)
            % Generate blocks
            blockin  = [testpath N_test(j).name];
            blockout = [testpath 'dir_' N_test(j).name]; 
            mkdir(blockout);

            CHM_blockGenTimer = tic;
            CHM_blockGen(BlockSize,blockin,blockout);
    
            fileste = dir([blockout '/*.png']); 

            fprintf('Running CHM_blockGen on %s generated %d blocks and took %f seconds\n\n',N_test(j).name,length(fileste),toc(CHM_blockGenTimer));

            str = [savingpath 'output_testImages/dir_' N_test(j).name];
            mkdir(str);
            outputs{length(fileste)} = [];
            parfor i = 1:length(fileste)
                testCHMTimer = tic;
                img = imread([blockout '/' fileste(i).name]);
                clabels = testCHM(img,savingpath,param);
                outputs{i} = clabels;
                fprintf('Running testCHM # %d of %d on %s took %f seconds\n',i,length(fileste),fileste(i).name,toc(testCHMTimer));
            end

            fprintf('\n');
    
            for i = 1:length(fileste)
                clabels = outputs{i};
                imwrite(clabels,[str '/' fileste(i).name]);
            end
    
            clear fileste clabels outputs

            CHM_blockCombineTimer = tic;

            CHM_blockCombine(str,combineOutFile);
            fprintf('Running CHM_blockCombine on %s to generate %s took %f seconds\n\n',str,combineOutFile,toc(CHM_blockCombineTimer));
            rmdir(blockout,'s');
        end % for j = 1:numel(N_test)

        fprintf('Running TrainScript_testBlocks on %d input slices took %f seconds\n\n',length(N_test),toc(TrainScript_testBlocksTimer));

    catch err
        fprintf('Caught Fatal Exception: %s\n',err.getReport());
        exit(1);
    end

    exit(0);

end % function TrainScript_testBlocks

