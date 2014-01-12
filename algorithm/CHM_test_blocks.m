function CHM_test_blocks(input_files,outputpath,blocksize,bordersize,savingpath)
if nargin < 3 || nargin > 5; error('CHM_test_blocks must have 3 to 5 input arguments'); end
if nargin < 4; bordersize = 0; end
if nargin < 5; savingpath = fullfile('.', 'temp'); end

param = load(fullfile(savingpath, 'param'), 'Nfeatcontext', 'Nlevel', 'Nstage');

% Add path to functions required for feature extraction.
[my_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(my_path, 'FilterStuff')));

files_te  = GetInputFiles(input_files);
files_out = GetOutputFiles(outputpath, files_te);

bs = [blocksize-2*bordersize blocksize-2*bordersize];
brd = [bordersize bordersize];
proc = @(bs) ProcessBlock(bs, savingpath, param);

opened_pool = 0;
try; if usejava('jvm') && ~matlabpool('size'); matlabpool open; opened_pool = 1; end; catch ex; end;

for i = 1:length(files_te)
    [~,~,ext] = fileparts(files_te{i});
    if strcmpi(ext,'.tif') || strcmpi(ext,'.tiff')
        info = imfinfo(files_te{i});
        w = info.Width;
        h = info.Height;
        if mod(w, bs(2)) == 0 && mod(h, bs(1)) == 0;
             blockproc(files_te{i}, bs,proc, 'BorderSize',brd, 'UseParallel',true, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false, 'Destination',files_out{i});
             continue;
        end
        im = blockproc(files_te{i}, bs,proc, 'BorderSize',brd, 'UseParallel',true, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false);
        im = im(1:h,1:w); % when reading directly from file the image is always padded with zeros to a multiple of the block size size.
    else
        im = blockproc(imread(files_te{i}),bs,proc, 'BorderSize',brd, 'UseParallel',true, 'PadPartialBlocks',true, 'PadMethod','symmetric', 'TrimBorder',false);
    end
    imwrite(im, files_out{i});
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
    %output = block_struct.data/255.0;
    
    % Remove border and partial padding
    sz = brd + 1 + min([block_struct.imageSize-block_struct.location;size(output)-2*brd-1]);
    output = uint8(output(brd(1)+1:sz(1), brd(2)+1:sz(2))*255);
