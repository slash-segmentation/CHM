function im=FillBackgroundWithReflection(im,t,l,b,r)
% Fills the 'background' of the image with a reflection of the foreground.
% The foreground is given by the top-left and bottom-right corners (such that im(t:b,l:r) is the foreground).
% Currently the foreground must be wider or taller than the background.
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
im(1:t-1,:) = im(t+t-2:-1:t,:);
im(:,1:l-1) = im(:,l+l-2:-1:l);
im(b+1:end,:) = im(b:-1:b+b-end+1,:);
im(:,r+1:end) = im(:,r:-1:r+r-end+1);
