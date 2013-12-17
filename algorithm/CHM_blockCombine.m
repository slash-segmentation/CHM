function CHM_blockCombine( BlockDir, OutFile )

Imgs_block = dir([BlockDir '/block*.png']);

for i = 1:numel(Imgs_block)
    Block = imread([BlockDir '/' Imgs_block(i).name]);
    delim = regexp(Imgs_block(i).name,'[_ .]');
    S(i,1) = str2num(Imgs_block(i).name(delim(1)+1:delim(2)-1));
    S(i,2) = str2num(Imgs_block(i).name(delim(2)+1:delim(3)-1));
    [S(i,3) S(i,4)] = size(Block);
end

numBlkH = numel(unique(S(:,1)));
numBlkW = numel(unique(S(:,2)));
dimW = sum(S(1:numBlkW,4));
dimH = sum(S(1:numBlkW:numel(Imgs_block),3));

I = zeros(dimH,dimW);

R = 1; R0 = 1;
for i = 1:numel(Imgs_block)
    Block = imread([BlockDir '/' Imgs_block(i).name]);
    C = S(i,2);
    RF = S(i,1);
    if RF ~= R0
        R = R + S(i-1,3);
        R0 = S(i,1);
    end
    if C == 1
        C0 = 1;
    end        
    CF = C0 + S(i,4) - 1;
    S(i,5) = R;
    S(i,6) = S(i,5) + S(i,3) - 1;
    S(i,7) = C0;
    S(i,8) = CF;
    C0 = CF +1;
    I(S(i,5):S(i,6),S(i,7):S(i,8)) = Block;
end

imwrite(uint8(I),OutFile);

delete([BlockDir '/block*.png']);

end
