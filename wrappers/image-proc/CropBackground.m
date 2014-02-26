function im=CropBackground(im,t,l,b,r)
% Removes the 'background' of the image.
% The background is given by the sizes of the top, left, bottom, and right padding already on the image.
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im = im(t+1:end-b,l+1:end-r);
