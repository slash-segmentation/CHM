function [t,l,b,r]=GetForegroundArea(im,bg)
% Get the area of the foreground, given by the top-left and bottom-right corners (such that im(t:b,l:r) is the foreground).
% If bg is not given, it is calculated from the edges of the image.
% The image can be a string or a matrix.
if nargin < 1 || nargin > 2; error('TrimBackground must have 1 or 2 input arguments'); end;
if ischar(im); im = imread(im); end;
t = 1; l = 1; b = size(im,1); r = size(im,2);
if nargin == 1;
  % Calculate bg color using solid strips on top, bottom, left, or right
  % Instead of range() could use all(im(1,:) == im(1,1))
  if range(im(1,:)) == 0 || range(im(:,1)) == 0;
    bg = im(1,1);
  elseif range(im(end,:)) == 0 || range(im(:,end)) == 0;
    bg = im(end,end);
  else;
    % no discoverable bg color, return the entire image
    return;
  end;
end;
sz = size(im);
while t < sz(1) && all(im(t,:) == bg); t = t+1; end;
while b > t     && all(im(b,:) == bg); b = b-1; end;
while l < sz(2) && all(im(:,l) == bg); l = l+1; end;
while r > l     && all(im(:,r) == bg); r = r-1; end;
