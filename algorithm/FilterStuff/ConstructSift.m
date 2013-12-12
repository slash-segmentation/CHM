function siftArr = ConstructSift(I)


im1=imfilter(I,fspecial('gaussian',7,1.),'same','replicate');
I=im2double(im1);
gridSpacing = 1;

% patchSize = 24; %must be even
% %Padd the input image
% I_Padd = zeros( size(I,1)+patchSize-2 , size(I,2)+patchSize-2 );
% I_Padd(patchSize/2:end-patchSize/2+1,patchSize/2:end-patchSize/2+1) = I;
% siftArr1=dense_sift(I_Padd,patchSize,gridSpacing);
% 
% patchSize = 8; %must be even
% %Padd the input image
% I_Padd = zeros( size(I,1)+patchSize-2 , size(I,2)+patchSize-2 );
% I_Padd(patchSize/2:end-patchSize/2+1,patchSize/2:end-patchSize/2+1) = I;
% siftArr2=dense_sift(I_Padd,patchSize,gridSpacing);

patchSize = 16; %must be even
%Padd the input image
I_Padd = zeros( size(I,1)+patchSize-2 , size(I,2)+patchSize-2 );
I_Padd(patchSize/2:end-patchSize/2+1,patchSize/2:end-patchSize/2+1) = I;
siftArr3=dense_sift(I_Padd,patchSize,gridSpacing);
siftArr = double(siftArr3');

% 
% siftArr = max(siftArr1,siftArr2);
% siftArr = max(siftArr,siftArr3);
% siftArr = double(siftArr');

