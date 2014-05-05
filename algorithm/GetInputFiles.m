function output = GetInputFiles(x);
% Gets the files to use. Always returns a 1-by-n cell-array of currently valid, existing, file names.
% The argument can be a cell str array already, in which case it is filtered for existing files and returned. 
% Or the argument can be a char array like one of the following or a combination seperated by commas:
%  * path to a folder            - all PNGs and TIFFs in that folder
%  * path to a file              - only that file
%  * path with numerical pattern - get all files matching the pattern (pattern must have #s in it and end with a ;5-15 or other numbers, the #s are replaced by the values at the end with leading zeros as necessary)
%  * path with wildcard pattern  - get all files matching the pattern (* in the pattern means any number of any characters)
if iscellstr(x);
    output = cellfun(@(f) exist(f, 'file'), x);
    bad    = x(output~=2);
    if length(bad) ~= 0;
        for i = 1:length(bad); disp(['No such file "' bad{i} '"']); end;
        output = x(output==2);
    else;
        output = x(:);
    end;
elseif ischar(x) && isvector(x);
    [d,f,e] = fileparts(x);
    if strcmp(d, ''); type = exist(fullfile('.', x), 'file'); # Workaround for a bug in MATLAB when `file` is 'input' or some other built-in names
    else; type = exist(x, 'file'); end;
    if type == 7;
        % is a directory, grab all PNGs and TIFFs there
        files = dir(x);
        output = {};
        for i = 1:length(files);
            f = files(i).name;
            [~, ~, ext] = fileparts(f);
            if any(strcmp(lower(ext), {'.png', '.tif', '.tiff'})); output = [output,fullfile(x,f)]; end;
        end;
        if length(output) == 0; disp(['No PNG or TIFF files found in "' x '"']); end;
    elseif type == 2;
        % is a single file
        output = {x};
    else % if type == 0;
        % does not exist, maybe it is a multiple listing, numerical pattern, or wildcard pattern
        output = get_files_multiple_listing(x); if iscellstr(output); output = unique(output); return; end;
        output = get_files_numerical(x);        if iscellstr(output); output = unique(output); return; end;
        output = get_files_wildcard(x);         if iscellstr(output); output = unique(output); return; end;
        disp(['No such file "' x '"']);
        output = cell(1,0);
    end
else
    err = MException('GetInputFiles:ArgumentTypeException', 'Argument to GetInputFiles was not a usable type');
    throw(err);
end
output = unique(output);


function output = get_files_multiple_listing(x);
    files = regexp(x, ',', 'split'); % only in newer versions of MATLAB: strsplit(x, ',')
    if length(files) <= 1; output = 0; return; end;
    output = {};
    for i = 1:length(files);
        if length(files(i)) ~= 0;
            output = [output,GetInputFiles(files{i})];
        end;
    end;

function output = get_files_numerical(x);
    tokens = regexp(x, '^([^#]*)(#+)([^#;]*);(\d+)-(\d+)$', 'tokens');
    if length(tokens) ~= 1; output = 0; return; end;
    a = tokens{1}{1};
    digits = length(tokens{1}{2});
    b = tokens{1}{3};
    lower = str2num(tokens{1}{4});
    upper = str2num(tokens{1}{5});
    if upper < lower; output = 0; return; end;
    
    output = cell(1,(upper-lower+1));
    pattern = [a '%0' num2str(digits) 'd' b];
    n = 0;
    for i = lower:upper;
        f = sprintf(pattern, i);
        if exist(f, 'file') ~= 2;
            disp(['No such file "' f '"']);
        else;
            n = n + 1;
            output{n} = f;
        end;
    end;
    output = output(1:n);

function output = get_files_wildcard(x);
    if length(strfind(x, '*')) == 0; output = 0; return; end;
    [folder, ~, ~] = fileparts(x);
    files = dir(x);
    n = length(files);
    if n == 0; output = 0; return; end;
    output = cell(1,n);
    for i = 1:n; output{i} = fullfile(folder, files(i).name); end;
