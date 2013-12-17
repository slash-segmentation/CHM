function CHM_blockGen( BlockSize, file, outdir )

I = imread(file);

height = size(I,1);
width = size(I,2);

numBlkH = ceil(height / BlockSize);
numBlkW = ceil(width / BlockSize);

fprintf('Image %s has dimensions of %dx%d and will be broken into %d tiles\n',file,width,height,(numBlkH*numBlkW));

[imgH,imgW,~] = size(I);
szBlkH = [repmat(fix(imgH/numBlkH),1,numBlkH-1) imgH-fix(imgH/numBlkH)*(numBlkH-1)];
szBlkW = [repmat(fix(imgW/numBlkW),1,numBlkW-1) imgW-fix(imgW/numBlkW)*(numBlkW-1)];

C = mat2cell(I,szBlkH,szBlkW)';

for i = 1:size(C,2)
    for j = 1:size(C,1)
        str = ['block_' sprintf('%03d',i) '_' sprintf('%03d',j) '.png'];
        imwrite(uint8(C{j,i}),[outdir '/' str]);
    end
end

