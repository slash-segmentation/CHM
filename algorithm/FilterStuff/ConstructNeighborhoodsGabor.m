function fvv = ConstructNeighborhoodsGabor(I)
sigma = 2:6;
lambdaFact = 2:.25:2.5;
orient = pi/6:pi/6:2*pi;
phase = [0 pi/2];
gama = 1;

featSize = numel(sigma)*numel(lambdaFact)*numel(orient)*numel(phase)*numel(gama);
fv = zeros(featSize,numel(I));
co = 1;
for i = sigma
    for j = gama
        for k = orient
            for m = lambdaFact
                for l = phase
                    GW =  gabor_fn (i,k,m*i,l,j);
%                     figure(1),imagesc(GW),title(['sig = ' num2str(i), ...
%                         'gam = ' num2str(j), 'orient = ' num2str(k)]);
                    temp = im2double(imfilter(I,GW,'symmetric'));
                    fv(co,:) = reshape(temp,1,numel(I));
                    co = co + 1;
                end
            end
        end
    end
end


fvv = zeros(featSize/2,numel(I));

for i = 1:featSize/2
    temp1 = fv(2*i-1,:);
    temp2 = fv(2*i,:);
    fvv(i,:) = sqrt(temp1.^2 + temp2.^2);
end


% 
% sigma = 2:4;
% lambdaFact = 2:.25:2.5;
% orient = pi/4:pi/4:2*pi;
% phase = [0 pi/2];
% gama = 1;
% 
% featSize = numel(sigma)*numel(lambdaFact)*numel(orient)*numel(phase)*numel(gama);
% fv = zeros(featSize,numel(I));
% co = 1;
% for i = sigma
%     for j = gama
%         for k = orient
%             for m = lambdaFact
%                 for l = phase
%                     GW =  gabor_fn (i,k,m*i,l,j);
% %                     figure(1),imagesc(GW),title(['sig = ' num2str(i), ...
% %                         'gam = ' num2str(j), 'orient = ' num2str(k)]);
%                     temp = im2double(imfilter(I,GW,'symmetric'));
%                     fv(co,:) = reshape(temp,1,numel(I));
%                     co = co + 1;
%                 end
%             end
%         end
%     end
% end
% 
% 
% fvv = zeros(featSize/2,numel(I));
% 
% for i = 1:featSize/2
%     temp1 = fv(2*i-1,:);
%     temp2 = fv(2*i,:);
%     fvv(i,:) = sqrt(temp1.^2 + temp2.^2);
% end
% 






