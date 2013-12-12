function [feat NVI] = ConstructEdgeFilt(I)
w = im2double(I);
%parameters
Nx1=7;Sigmax1=1;Nx2=7;Sigmax2=1;Theta1=pi/2;
Ny1=7;Sigmay1=1;Ny2=7;Sigmay2=1;Theta2=0;  

filterx=d2dgauss(Nx1,Sigmax1,Nx2,Sigmax2,Theta1);
Ix= conv2(w,filterx,'same');

filtery=d2dgauss(Ny1,Sigmay1,Ny2,Sigmay2,Theta2);
Iy=conv2(w,filtery,'same'); 

NVI=sqrt(Ix.*Ix+Iy.*Iy);


offset = SquareNeighborhood(3);
NVII = padReflect(NVI,3);
feat = ConstructNeighborhoods(NVII,offset,0);









% Code from here:http://www.bic.mni.mcgill.ca/~joubin/mtrr/MATLAB_CODES/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function "d2dgauss.m":
% This function returns a 2D edge detector (first order derivative
% of 2D Gaussian function) with size n1*n2; theta is the angle that
% the detector rotated counter clockwise; and sigma1 and sigma2 are the
% standard deviation of the gaussian functions.
function h = d2dgauss(n1,sigma1,n2,sigma2,theta)
r=[cos(theta) -sin(theta);
   sin(theta)  cos(theta)];
h = zeros(n2,n1);
for i = 1 : n2 
    for j = 1 : n1
        u = r * [j-(n1+1)/2 i-(n2+1)/2]';
        h(i,j) = gauss(u(1),sigma1)*dgauss(u(2),sigma2);
    end
end
h = h / sqrt(sum(sum(abs(h).*abs(h))));

% Function "gauss.m":
function y = gauss(x,std)
y = exp(-x^2/(2*std^2)) / (std*sqrt(2*pi));

% Function "dgauss.m"(first order derivative of gauss function):
function y = dgauss(x,std)
y = -x * gauss(x,std) / std^2;
     
     
     
     
     
     
     
     
     
     
