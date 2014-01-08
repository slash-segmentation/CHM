function CHM_test_blocks(input_files,outputpath,blocksize,bordersize,savingpath)
if nargin < 3 || nargin > 5; error('CHM_test_blocks must have 3 to 5 input arguments'); end
if nargin < 4; bordersize = 0; end
if nargin < 5; savingpath = fullfile('.', 'temp'); end

CHM_test_blocks_timer = tic;

param = load(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage');

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

opened_pool = 0;
try; if ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

files_te = GetFiles(input_files);
if exist(outputpath,'file')~=7; mkdir(outputpath); end

bs = [blocksize-2*bordersize blocksize-2*bordersize];
brd = [bordersize bordersize];
proc = @(bs) ProcessBlock(bs, savingpath, param);

for i = 1:length(files_te)
    [folder,filename,ext] = fileparts(files_te{i});
    output_file = fullfile(outputpath, [filename ext]);
    if ext=='.tif'
                blockproc(       files_te{i}, bs,proc, 'BorderSize',brd, 'UseParallel',true, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false, 'Destination',output_file);
    else
        imwrite(blockproc(imread(files_te{i}),bs,proc, 'BorderSize',brd, 'UseParallel',true, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false),              output_file);
    end
end

if opened_pool; matlabpool close; end

fprintf('CHM_test_blocks on %d input slices took %f seconds\n\n',length(files_te),toc(CHM_test_blocks_timer));
end

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

end
