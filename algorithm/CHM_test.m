function CHM_test(input_files,outputpath,blocksize,bordersize,savingpath,tiles_to_proc,hist_eq)
% CHM_test   CHM Image Testing Phase Script
%   CHM_test(input_files, outputpath, [blocksize='auto'], [bordersize=[0 0]], [savingpath='./temp'], [tiles_to_proc=[]], [hist_eq=true])
%
%   input_files is a set of data files to test, see below for details.
%   outputpath is the folder to save the generated images to.
%       The images will have the same name and type as the input files but be placed in this folder.
%   blocksize is the size of the blocks to process, which should be identical to the training image sizes.
%       By default this reads the information from the param.mat file.
%   bordersize is the amount of overlap between blocks to discard.
%       By default a 50x50, but 25-75px overlap is recommended (will depend on whats being segmented).
%   savingpath is the folder to save the trained model to (along with temporary files)
%       Only need to keep MODEL_level#_stage#.mat and param.mat files in that folder.
%   tiles_to_proc is a list of tiles to process.
%       By default all tiles are processed.
%   hist_eq is either true or false if the testing data should be histogram-equalized to the training data or not.
%       By default it is true if the model includes the histogram to equalize to.
%
% input_files is a comma-seperated list of the following:
%   path to a folder            - all PNGs in that folder
%   path to a file              - only that file 
%   path with numerical pattern - get all files matching the pattern
%       pattern must have #s in it and end with a semicolon and number range
%       the #s are replaced by the values at the end with leading zeros
%       example: in/####.png;5-15 would do in/0005.png through in/0015.png
%   path with wildcard pattern  - get all files matching the pattern
%       pattern has * in it which means any number of any characters
%       example: in/*.tif does all TIFF images in that directory

if nargin < 2 || nargin > 7; error('CHM_test must have 2 to 7 input arguments'); end
if nargin < 3; blocksize = 'auto'; end
if nargin < 4; bordersize = 50; end
if nargin < 5; savingpath = fullfile('.', 'temp'); end
if nargin < 6; tiles_to_proc = []; end
if nargin < 7; hist_eq = true; end

if ~ismcc && ~isdeployed
    % Add path to functions required for feature extraction (already included in compiled version)
    [my_path, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(my_path, 'FilterStuff')));
else
    % Parse non-string arguments
    blocksize  = ParseArgument(blocksize);
    bordersize = ParseArgument(bordersize);
    tiles_to_proc = ParseArgument(tiles_to_proc);
    hist_eq = ParseArgument(hist_eq);
end

param_vars = whos('-file', fullfile(savingpath, 'param.mat'))
param = load(fullfile(savingpath, 'param.mat'), 'Nfeatcontext', 'Nlevel', 'Nstage');
if ischar(blocksize) && strcmpi(blocksize, 'auto')
    if ~ismember('TrainingSize', {param_vars.name}); error('''auto'' was specified for blocksize but model does not contain the training image size'); end;
    pts = load(fullfile(savingpath, 'param.mat'), 'TrainingSize');
    blocksize = pts.TrainingSize;
end
my_imread = @imread
if hist_eq
    if ismember('hgram', {param_vars.name})
        pts = load(fullfile(savingpath, 'param.mat'), 'hgram');
        hgram = pts.hgram;
        my_imread = @(fn) histeq(imread(fn), hgram);
    else
        fprintf('Warning: training data histogram not included in model, make sure you manually perform histogram equalization on the testing data.\n');
    end
end

files_te  = GetInputFiles(input_files);
files_out = GetOutputFiles(outputpath, files_te);

if numel(bordersize) == 1; brd = [bordersize bordersize]; elseif numel(bordersize) == 2; brd = bordersize(:)'; else; error('bordersize argument to CHM_test must have 1 or 2 elements'); end
if numel(blocksize)  == 1; bs  = [blocksize  blocksize ]; elseif numel(blocksize)  == 2; bs  = blocksize (:)'; else;  error('blocksize argument to CHM_test must have 1 or 2 elements'); end
bs = bs-2*brd;
if ndims(tiles_to_proc) == 2 && size(tiles_to_proc, 2) == 2 && size(tiles_to_proc, 1) ~= 0
    locs_to_proc = repmat(bs, [size(tiles_to_proc, 1) 1]).*(tiles_to_proc-1)+1;
    proc = @(block_struct) ProcessBlock_LocOnly(block_struct, savingpath, param, locs_to_proc);
elseif all(size(tiles_to_proc) ~= 0)
    error('tiles_to_proc argument to CHM_test must be an N by 2 matrix');
else
    proc = @(block_struct) ProcessBlock(block_struct, savingpath, param);
end
%args = {'BorderSize',brd, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false, 'UseParallel',true};

opened_pool = 0;
try; if usejava('jvm') && ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

for i = 1:length(files_te)
    % Disabling all TIFF-specific code for now. Maybe one day we will re-add it.
    % It has benefits of better memory usage and runs twice as fast when running in parallel ('UseParallel', true) but normally has a penalty of 2-5 seconds per tile (especially horrendous when running with selected tiles only - can add 30 min to a 10 min job).
    %[~,~,ext] = fileparts(files_te{i});
    %if strcmpi(ext,'.tif') || strcmpi(ext,'.tiff')
    %    info = imfinfo(files_te{i});
    %    w = info.Width;
    %    h = info.Height;
    %    if mod(w, bs(2)) == 0 && mod(h, bs(1)) == 0;
    %         blockproc(files_te{i}, bs,proc, args{:}, 'Destination',files_out{i});
    %         continue;
    %    end
    %    im = blockproc(files_te{i}, bs,proc, args{:});
    %    im = im(1:h,1:w); % when reading directly from file the image is always padded with zeros to a multiple of the block size size.
    %else
    %    im = blockproc(imread(files_te{i}),bs,proc, args{:});
    %end
    %imwrite(im, files_out{i});

    imwrite(blockproc(my_imread(files_te{i}),bs,proc, 'BorderSize',brd, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false, 'UseParallel',true), files_out{i});
end

if opened_pool; matlabpool close; end


function output=ProcessBlock(block_struct, savingpath, param)
    % Designed to remove the extra padding added to the right and bottom blocks
    
    %block_struct.border:    A two-element vector, [V H], that specifies the size of the vertical and horizontal padding around the block of data. (See the 'BorderSize' parameter in the Inputs section.)
    %block_struct.blockSize: A two-element vector, [rows cols], that specifies the size of the block data. If a border has been specified, the size does not include the border pixels.
    %block_struct.data:      M-by-N or M-by-N-by-P matrix of block data
    %block_struct.imageSize: A two-element vector, [rows cols], that specifies the full size of the input image.
    %block_struct.location:  A two-element vector, [row col], that specifies the position of the first pixel (minimum-row, minimum-column) of the block data in the input image. If a border has been specified, the location refers to the first pixel of the discrete block data, not the added border pixels.
    
    brd = block_struct.border;
    
    %fprintf('Processing block at [%4d %4d] of size [%3d %3d] with border [%3d %3d] out of [%4d %4d]\n', ...
    %    block_struct.location(1), block_struct.location(2), size(block_struct.data,1)-2*brd(1), size(block_struct.data,2)-2*brd(2), brd(1), brd(2), block_struct.imageSize(1), block_struct.imageSize(2));
    
    % Process block
    output = testCHM(block_struct.data, savingpath, param);
    
    % Remove border and partial padding
    sz = brd + 1 + min([block_struct.imageSize-block_struct.location;size(output)-2*brd-1]);
    output = uint8(output(brd(1)+1:sz(1), brd(2)+1:sz(2))*255);


function output=ProcessBlock_LocOnly(block_struct, savingpath, param, locs_to_proc)
    % Only processes when location is in locs_to_proc, otherwise just returns 0s
    loc = block_struct.location;
    if any(all(repmat(loc,size(locs_to_proc,1),1)==locs_to_proc,2)) % equivilent to but 10x faster then: any(ismember(locs_to_proc,loc,'rows'))
      output = ProcessBlock(block_struct, savingpath, param);
    else
      output = zeros(min([block_struct.imageSize-loc+1;size(block_struct.data)-2*block_struct.border]),'uint8');
    end
