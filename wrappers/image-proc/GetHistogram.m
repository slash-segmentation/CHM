function counts=GetHistogram(im)
% Gets the histogram of the image as the counts seen of each pixel value (offset 1 is pixel value 0, offset 256 is pixel value 255).
% The image can be a string or a matrix.
% If a cell-array of matrices or strings is given, the histogram is summed across all images.
if iscell(im);
    total_counts = zeros(256);
    for i in numel(im)
        [counts,x]=imhist(im{i});
        total_counts = total_counts + counts;
    end
else
    if ischar(im); im = imread(im); end;
    [counts,x]=imhist(im);
end
