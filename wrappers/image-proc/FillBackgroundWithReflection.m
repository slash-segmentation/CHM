function im=FillBackgroundWithReflection(im,t,l,b,r)
% Fills the 'background' of the image with a reflection of the foreground.
% The background is given by the sizes of the top, left, bottom, and right padding already on the image.
% Currently the foreground must be wider and taller than the background.
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im(1:t,:) = im(2*t:-1:t+1,:);
im(:,1:l) = im(:,2*l:-1:l+1);
im(end-b+1:end,:) = im(end-b:-1:end-2*b+1,:);
im(:,end-r+1:end) = im(:,end-r:-1:end-2*r+1);
