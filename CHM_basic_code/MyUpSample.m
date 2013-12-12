function output = MyUpSample(I,L)
% temp = repmat(reshape(I,[1,numel(I)]),2,1);
% I = reshape(temp,2*size(I,1),size(I,2));
% I = I';
% temp = repmat(reshape(I,[1,numel(I)]),2,1);
% output = reshape(temp,2*size(I,1),size(I,2))';

% supports both 2d and 3d upsampling

if L==0, output = I; return; end

temp = zeros(size(I,1),2*size(I,2),size(I,3));
temp(:,1:2:end,:) = I;
temp(:,2:2:end,:) = I;

I = zeros(2*size(I,1),2*size(I,2),size(I,3));
I(1:2:end,:,:) = temp;
I(2:2:end,:,:) = temp;

output = MyUpSample(I,L-1);
