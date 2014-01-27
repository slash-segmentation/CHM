function counts=GetHistogram(im)
% Gets the histogram of the image as the counts seen of each pixel value (offset 1 is pixel value 0, offset 256 is pixel value 255).
% The image can be a string or a matrix.
% If a cell-array of matrices or strings is given, the histogram is summed across all images.
if iscell(im);
    if numel(im) < 1; counts = zeros(256,1); return
    [counts,~] = imhist(im{1});
    for i in 2:numel(im)
        [c,~] = imhist(im{i});
        counts = counts + c;
    end
else
    if ischar(im); im = imread(im); end;
    [counts,~] = imhist(im);
end
