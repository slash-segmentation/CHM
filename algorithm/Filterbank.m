function feat = Filterbank(I)

haar = ConstructHaar(I);
Hog = ConstructHOG(I);
Gab = ConstructNeighborhoodsGabor(I);
SifT = ConstructSift(I);
offset = StencilNeighborhood(7);
I = im2double(I);
I = padReflect(I,7);
intenfeat = ConstructNeighborhoods(I,offset,0);
feat = [haar ; Hog ; Gab ; SifT;intenfeat];