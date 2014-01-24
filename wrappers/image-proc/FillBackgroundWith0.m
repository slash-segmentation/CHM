function im=FillBackgroundWith0(im,t,l,b,r)
% Fills the 'background' of the image with 0s.
% The foreground is given by the top-left and bottom-right corners (such that im(t:b,l:r) is the foreground).
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im(1:t-1,:) = 0;
im(:,1:l-1) = 0;
im(b+1:end,:) = 0;
im(:,r+1:end) = 0;
