function HaarArr = ConstructHaar(I)
I = im2double(I);
Ip = padReflect(I,8);
II = cc_cmp_II(Ip);

H1 = cc_Haar_features(II,16);
H1 = H1(1:size(I,1),1:size(I,2),:);
temp1 = reshape(H1(:,:,1),1,numel(I));
temp2 = reshape(H1(:,:,2),1,numel(I));
HaarArr = [temp1;temp2];
