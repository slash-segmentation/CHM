function im=CropBackground(im,t,l,b,r)
% Removes the 'background' of the image.
% The foreground is given by the top-left and bottom-right corners (such that im(t:b,l:r) is the foreground).
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im = im(t:b,l:r);
