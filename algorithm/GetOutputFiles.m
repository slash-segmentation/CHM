function output = GetOutputFiles(outputpath, inputs);
if iscellstr(outputpath)
    output = outputpath;
else
    if exist(outputpath,'file')~=7; mkdir(outputpath); end;
    output = cell(1,length(inputs));
    for i = 1:length(inputs)
        [~,filename,ext] = fileparts(inputs{i});
        output{i} = fullfile(outputpath, [filename ext]);
    end
end