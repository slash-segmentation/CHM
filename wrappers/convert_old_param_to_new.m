function convert_old_param_to_new(model_folder, Nstage)
if nargin < 1; model_folder = fullfile('.', 'temp'); end
if nargin < 2
    % calculate Nstage from other files
    Nstage=0;
    files = dir(fullfile(model_folder, 'MODEL_level0_stage*.mat'));
    for i = 1:length(files)
        if files(i).isdir; continue; end;
        NS = sscanf(files(i).name, 'MODEL_level0_stage%u.mat');
        if NS > Nstage; Nstage = NS; end;
    end
    disp(sprintf('Nstage of %d obtained from filenames',Nstage));
end

param = load(fullfile(model_folder, 'param'), 'param');
Nlevel = param.param.Nlevel;
Nfeatcontext = param.param.Nfeatcontext;
save(fullfile(model_folder, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage', '-v7.3');
