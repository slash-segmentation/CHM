function im=HistogramEqualize(im,histogram)
% The image can be a string or a matrix.
if nargin < 1 || nargin > 2; error('HistogramEqualize must have 1 to 2 input arguments'); end
if ischar(im); im = imread(im); end;
if nargin==2;
    im=histeq(im,histogram);
else;
    im=histeq(im);
end;
