function output = MyDownSample(I,L)

if L==0, output = I; return; end
 
    [nr nc ni] = size(I);
    if mod(nr,2)~=0
        I = [I;I(end,:,:)];
        [nr nc ni] = size(I);
    end
    if mod(nc,2)~=0
        I = [I I(:,end,:)];
        [nr nc ni] = size(I);
    end
    I = imresize(I,[nr/2 nc/2]); 
    output = MyDownSample(I,L-1);