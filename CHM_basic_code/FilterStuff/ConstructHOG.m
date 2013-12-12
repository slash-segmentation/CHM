function HOGArr = ConstructHOG(I)
I = im2double(I);
offset = SquareNeighborhood(7);
II = padReflect(I,7);
fv = ConstructNeighborhoods(II,offset,0);
fv = reshape(fv,[15 15 numel(I)]);
HOGArr = zeros(36,numel(I));
for i = 1:size(fv,3)
    HOGArr(:,i) = HoG(fv(:,:,i));
end