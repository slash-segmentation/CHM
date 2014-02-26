function im_out=AddBackground(im,t,l,b,r)
% Enlarges an image by adding 0s to the edge.
% The amount of background to add is given by the sizes of the top, left, bottom, and right padding.
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
sz = size(im);
im_out = zeros(t+sz(1)+b,l+sz(2)+r,class(im)); %,'like',im);
im_out(t+1:t+sz(1),l+1:l+sz(2)) = im; % im_out(t+1:end-b,l+1:end-r) = im;
