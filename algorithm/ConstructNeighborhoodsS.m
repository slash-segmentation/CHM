function fv = ConstructNeighborhoodsS(I)

% I = I(:,:,2);
offset = StencilNeighborhood(7);
[m n p] = size(I);
fv = zeros(numel(offset)/2 * p,m*n);
ro = numel(offset)/2;
for i = 1:size(I,3)
    II = padReflect(I(:,:,i),7);
    fv((i-1)*ro+1:i*ro,:) = ConstructNeighborhoods(II,offset,0);
end

