function CHM_dump_param(savingpath)
if nargin < 0 || nargin > 1; error('CHM_dump_param must have 0 to 1 input arguments'); end
if nargin < 0; savingpath = fullfile('.', 'temp'); end

param = load(fullfile(savingpath, 'param.mat'));
fprintf('Nfeatcontext: %d\n', param.Nfeatcontext);
fprintf('Nlevel: %d\n', param.Nlevel);
fprintf('Nstage: %d\n', param.Nstage);
if isfield(param,'TrainingSize');
    fprintf('TrainingSize: %d x %d (HxW)\n', param.TrainingSize(1), param.TrainingSize(2));
else
    fprintf('TrainingSize: not available\n');
end
if isfield(param,'hgram');
    fprintf(['hgram:' num2str(h') '\n']);
else
    fprintf('hgram: not available\n');
end
