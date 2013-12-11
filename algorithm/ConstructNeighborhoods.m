function fv = ConstructNeighborhoods (I, offset, flag)
%fv = ConstructNeighborhoods (I, offset, flag)
%
% FLAG indicates if the neighborhoods should be fully contained inside the
% image (FLAG=0), or the padded image can be considered (FLAG=1 or 2)
% FLAG=1 means that the extreme ends of the image are copied, and
% FLAG=2 means the image is simply padded with zeros

% derive settings
width = size(I,2);
height = size(I,1);
halfsize = ceil(max(abs(offset(:))));
fullsize = 2*halfsize + 1;
padsize = 2*halfsize;
offset_x = offset(1,:)';
offset_y = offset(2,:)';
dim = size(offset,2);

% construct padded image
Ipad = zeros(height + padsize, width + padsize);
Ipad(halfsize+1:halfsize+height,halfsize+1:halfsize+width) = I;
if (flag < 2)
	for k = 1:halfsize
		% copy first line to top padding area
	    Ipad(k,:) = Ipad(halfsize+1,:);
		% copy last line to bottom padding area
	    Ipad(height+padsize+1-k,:) = Ipad(halfsize+height,:);
	end;
	for k = 1:halfsize
		% similar to previous for loop, but for first and last columns
	    Ipad(:,k) = Ipad(:,halfsize+1);
	    Ipad(:,width+padsize+1-k) = Ipad(:,halfsize+width);
	end;
end

if (flag)
	npix = width * height;
    pos_x = ones(height,1)*(1:width);
    %pos_x = halfsize + bsxfun(@plus,reshape(pos_x,1,width*height),offset_x);
    pos_x = halfsize + (repmat(offset_x,1,npix) ...
			+ repmat(reshape(pos_x,1,npix),dim,1));
    pos_y = (1:height)'*ones(1,width);
    %pos_y = halfsize + bsxfun(@plus,reshape(pos_y,1,width*height),offset_y);
    pos_y = halfsize + (repmat(offset_y,1,npix) ...
			+ repmat(reshape(pos_y,1,npix),dim,1));
else
	npix = (width-padsize) * (height-padsize);
    pos_x = ones(height-padsize,1)*(halfsize+1:width-halfsize);
    %pos_x = halfsize + bsxfun(@plus,reshape(pos_x,1,(width-padsize)*(height-padsize)),offset_x);
    pos_x = halfsize + (repmat(offset_x,1,npix) ...
			+ repmat(reshape(pos_x,1,npix),dim,1));
    pos_y = (halfsize+1:height-halfsize)'*ones(1,width-padsize);
    %pos_y = halfsize + bsxfun(@plus,reshape(pos_y,1,(width-padsize)*(height-padsize)),offset_y);
    pos_y = halfsize + (repmat(offset_y,1,npix) ...
			+ repmat(reshape(pos_y,1,npix),dim,1));
end;
temp = height + padsize;
index = pos_y + (pos_x-1)*temp;
fv = Ipad(index);

