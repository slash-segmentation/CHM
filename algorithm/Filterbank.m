function feat = Filterbank(I)
if strcmpi(I, 'count'); feat = 6; return; end;

haar = ConstructHaar(I);
Hog = ConstructHOG(I);
Gab = ConstructNeighborhoodsGabor(I);
SifT = ConstructSift(I);
ED = ConstructEdgeFilt(I);
offset = StencilNeighborhood(10);
I = im2double(I);
I = padReflect(I,10);
intenfeat = ConstructNeighborhoods(I,offset,0);
feat = [haar ; Hog ; ED; Gab ; SifT;intenfeat];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
