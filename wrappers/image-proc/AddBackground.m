function im_out=AddBackground(im,t,l,h,w)
% The image can be a string or a matrix.
if ischar(im); im = imread(im); end;
sz = size(im);
im_out = zeros(h,w,class(im)); %,'like',im);
im_out(t:t+sz(1)-1,l:l+sz(2)-1) = im;
