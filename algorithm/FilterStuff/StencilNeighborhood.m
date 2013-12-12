function offset = StencilNeighborhood (radius)

halfsize = ceil(radius);
fullsize = 2*halfsize + 1;

offset_x = [];
for i = -halfsize:halfsize
	if (i == 0)
		offset_x(end + [1:2*halfsize+1],1) = [-halfsize:halfsize];
	else
		offset_x(end + [1:3],1) = [-abs(i), 0 , abs(i)];
	end
end

offset_y = [];
for i = -halfsize:halfsize
	if (i == 0)
		offset_y(end + [1:2*halfsize+1],1) = zeros(2*halfsize+1,1);
	else
		offset_y(end + [1:3],1) = i .* ones(3,1);
	end
end

offset_x = offset_x(:)';
offset_y = offset_y(:)';
offset = [offset_x; offset_y];
