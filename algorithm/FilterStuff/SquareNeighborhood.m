function offset = SquareNeighborhood (radius)

halfsize = ceil(radius);
fullsize = 2*halfsize + 1;
offset_x = ones(fullsize,1)*(-halfsize:halfsize);
offset_y = (-halfsize:halfsize)'*ones(1,fullsize);
offset_x = offset_x(:)';
offset_y = offset_y(:)';
offset = [offset_x; offset_y];

