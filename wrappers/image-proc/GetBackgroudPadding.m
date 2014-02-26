function [t,l,b,r]=GetBackgroundPadding(im,bg)
% Get the amount of the background padding, given by the number of rows/columns on the top, left, bottom, and right edges.
% If bg is not given, it is calculated from the edges of the image.
% The image can be a string or a matrix.
if nargin < 1 || nargin > 2; error('GetBackgroundPadding must have 1 or 2 input arguments'); end;
if ischar(im); im = imread(im); end;
sz = size(im);
if nargin == 1;
  % Calculate bg color using solid strips on top, bottom, left, or right
  % Instead of range() could use all(im(1,:) == im(1,1))
  if range(im(1,:)) == 0 || range(im(:,1)) == 0;
    bg = im(1,1);
  elseif range(im(end,:)) == 0 || range(im(:,end)) == 0;
    bg = im(end,end);
  else;
    % no discoverable bg color, return the entire image
    t = 0; l = 0; b = 0; r = 0;
    return;
  end;
end;
t = 1; l = 1; b = sz(1); r = sz(2);
while t < b && all(im(t,:) == bg); t = t+1; end;
while b > t && all(im(b,:) == bg); b = b-1; end;
while l < r && all(im(:,l) == bg); l = l+1; end;
while r > l && all(im(:,r) == bg); r = r-1; end;
t=t-1; b=sz(1)-b; l=l+1; r=sz(2)-r; % convert from corners to edges
