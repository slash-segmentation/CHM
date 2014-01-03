function output = GetFiles(x);
% Gets the files to use. Always returns a 1-by-n cell-array of currently valid, existing, file names.
% The argument can be a cell str array already, in which case it is filtered for existing files and returned. 
% Or the argument can be a char array like one of the following or a combination seperated by commas:
%  * path to a folder            - all PNGs in that folder
%  * path to a file              - only that file 
%  * path with numerical pattern - get all files matching the pattern (pattern must have #s in it and end with a ;5-15 or other numbers, the #s are replaced by the values at the end with leading zeros as necessary)
%  * path with wildcard pattern  - get all files matching the pattern (* in the pattern means any number of any characters)
if iscellstr(x);
    output = cellfun(@(f) exists(f, 'file'), x);
    bad    = x{output~=2};
    if length(bad) ~= 0;
        for i = 1:length(bad); disp ['No such file "' bad{i} '"']; end;
        output = x{output==2};
    else;
        output = x{:};
    end;
elseif ischar(x) && isvector(x);
    type = exists(x, 'file');
    if type == 7;
        % is a directory, grab all PNGs there
        pngs = dir(fullfile(x, '*.png'));
        n = length(pngs);
        disp ['No PNG files in "' x '"'];
        output = cell(1,n);
        for i = 1:n; output{i} = fullfile(x, pngs(i).name); end;
    elseif type == 2;
        % is a single file
        output = {x};
    else % if type == 0;
        % does not exist, maybe it is a multiple listing, numerical pattern, or wildcard pattern
        output = get_files_multiple_listing(x); if iscellstr(output); return; end;
        output = get_files_numerical(x);        if iscellstr(output); return; end;
        output = get_files_wildcard(x);         if iscellstr(output); return; end;
        disp ['No such file "' x '"'];
        output = {};
    end
else
    err = MException('GetFiles:ArgumentTypeException', 'Argument to GetFiles was not a usable type');
    throw(err);
end


function output = get_files_multiple_listing(x);
    files = strsplit(str,',')
    if length(files) <= 1; output = 0; return; end;
    output = {};
    for i = 1:length(files);
        if length(files(i)) ~= 0;
            output = [output,GetFiles(files{i})];
        end;
    end;

function output = get_files_numerical(x);
    tokens = regexp(x, '^([^#]*)(#+)([^#;]*);(\d+)-(\d+)$', 'tokens');
    if length(tokens) ~= 1: output = 0; return; end;
    a = tokens{1}{1};
    digits = length(tokens{1}{2});
    b = tokens{1}{3};
    lower = str2num(tokens{1}{4});
    upper = str2num(tokens{1}{5});
    if upper < lower; output = 0; return; end;
    
    output = cell{1,(upper-lower+1)};
    pattern = [a '%0' num2str(digits) 'd' b];
    n = 0;
    for i = lower:upper;
        f = sprintf(pattern, i);
        if exists(f, 'file') ~= 2;
            disp ['No such file "' f '"'];
        else;
            n = n + 1;
            output{n} = f;
        end;
    end;
    output = output{1:n};

function output = get_files_wildcard(x);
    if length(strfind(x, '*')) == 0; output = 0; return; end;
    [folder, ~, ~] = fileparts(x)
    pngs = dir(x);
    n = length(pngs);
    if n == 0; output = 0; return; end;
    output = cell(1,n)
    for i = 1:n; output{i} = fullfile(folder, pngs(i).name); end;
