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

blocksize = [blocksize-2*bordersize blocksize-2*bordersize];
bordersize = [bordersize bordersize];
if bordersize==0; proc = @(bs) testCHM(data,    savingpath, param);
else;             proc = @(bs) ProcessBlock(bs, savingpath, param); end;

for i = 1:length(files_te)
    [folder,filename,ext] = fileparts(files_te{i});
    output_file = fullfile(outputpath, [filename ext]);
    if ext=='.tif'
                blockproc(       files_te{i},  blocksize, proc, 'BorderSize', bordersize, 'UseParallel', true, 'Destination', output_file);
    else
        imwrite(blockproc(imread(files_te{i}), blocksize, proc, 'BorderSize', bordersize, 'UseParallel', true),               output_file);
    end

    %img = imread(files_te{i});

    %% Generate Blocks
    %GenBlocksTimer = tic;
    %blocks = GenBlocks(img, blocksize, bordersize);
    %fprintf('%s -> %d blocks [%f secs]\n\n',files_te{i},numel(blocks),toc(GenBlocksTimer));
    %clear img;
    
    %% Test CHM Blocks
    %TestCHMTimer = tic;
    %parfor j = 1:numel(blocks)
    %    blocks{j} = testCHM(blocks{j},savingpath,param);
    %end
    %fprintf('%s testing [%f secs]\n',files_te{i},toc(TestCHMTimer));

    % Combine Blocks
    %CombineBlocksTimer = tic;
    %out_filename = fullfile(outputpath, [filename ext]);
    %img = CombineBlocks(blocks, bordersize);
    %imwrite(img, out_filename);
    %fprintf('%d blocks -> %s [%f secs]\n\n',numel(blocks),out_filename,toc(CombineBlocksTimer));

    %clear img, blocks;
end

if opened_pool; matlabpool close; end

fprintf('CHM_test_blocks on %d input slices took %f seconds\n\n',length(files_te),toc(CHM_test_blocks_timer));
end

function output=ProcessBlock(block_struct, savingpath, param)
% Designed to remove the black border added by blockproc when bordersize != 0

%block_struct.border:    A two-element vector, [V H], that specifies the size of the vertical and horizontal padding around the block of data. (See the 'BorderSize' parameter in the Inputs section.)
%block_struct.blockSize: A two-element vector, [rows cols], that specifies the size of the block data. If a border has been specified, the size does not include the border pixels.
%block_struct.data:      M-by-N or M-by-N-by-P matrix of block data
%block_struct.imageSize: A two-element vector, [rows cols], that specifies the full size of the input image.
%block_struct.location:  A two-element vector, [row col], that specifies the position of the first pixel (minimum-row, minimum-column) of the block data in the input image. If a border has been specified, the location refers to the first pixel of the discrete block data, not the added border pixels.

data = block_struct.data;
loc = block_struct.location;
brd = block_struct.border;
bs = block_struct.blockSize - 2*brd; % not supposed to include border, but does
is = block_struct.imageSize;

%imwrite(data, sprintf('%04d_%04d_a.png', loc(1), loc(2)));

% Remove black border padding
if loc(1)==1; data=data(brd(1)+1:end,:); end;
if loc(2)==1; data=data(:,brd(2)+1:end); end;
if loc(1)+bs(1)>=is(1); data=data(1:end-brd(1),:); end;
if loc(2)+bs(2)>=is(2); data=data(:,1:end-brd(2)); end;

fprintf('Processing block at [%4d %4d] of size [%3d %3d] with border [%3d %3d] ([%3d %3d]) out of [%4d %4d]\n', ...
    loc(1), loc(2), bs(1), bs(2), brd(1), brd(2), size(data,1), size(data,2), is(1), is(2));

%imwrite(data, sprintf('%04d_%04d_b.png', loc(1), loc(2)));

% Process block
output = testCHM(data, savingpath, param);
%output = data;

% Add back removed border padding
if loc(1)==1; output = [zeros(brd(1),size(output,2)); output]; end;
if loc(2)==1; output = [zeros(size(output,1),brd(2))  output]; end;
if loc(1)+bs(1)>=is(1); output = [output; zeros(brd(1),size(output,2))]; end;
if loc(2)+bs(2)>=is(2); output = [output  zeros(size(output,1),brd(2))]; end;

%imwrite(output, sprintf('%04d_%04d_c.png', loc(1), loc(2)));
end
