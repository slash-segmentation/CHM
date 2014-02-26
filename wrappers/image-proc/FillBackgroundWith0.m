function im=FillBackgroundWith0(im,t,l,b,r)
% Fills the 'background' of the image with 0s.
% The background is given by the sizes of the top, left, bottom, and right padding already on the image.
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im(1:t,:) = 0;
im(:,1:l) = 0;
im(end-b+1:end,:) = 0;
im(:,end-r+1:end) = 0;
