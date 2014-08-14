function model = LearnAndOrNetMEX(varargin)
% function usage:
% [discriminants totalerror totalerrortest]= LearnAndOrNetMEX(xtrain, ytrain);
% OR [discriminants totalerror totalerrortest]= LearnAndOrNetMEX(xtrain, ytrain, xvalidation, yvalidation);
% OR [discriminants totalerror totalerrortest]= LearnAndOrNetMEX(xtrain, ytrain, options);
% OR [discriminants totalerror totalerrortest]= LearnAndOrNetMEX(xtrain, ytrain, xvalidation, yvalidation, options);
% Inputs:
%       xtrain: d by N training set ( d is number of attributes and N is
%               number of samples.
%       ytrain: 1 by N array of training labels.
%       xvalidation (optional): d by Nv validation set.
%       yvalidation (optional): 1 by Nv array of validation labels.
%       options:
%               options.epsilon = step size (default = 0.05).
%               options.momentum = momentum (default = 0.5).
%               options.maxepoch = maximum number of epochs (default = 25).
%               options.nGroup = Number of ORs (default = 15).
%               options.nDiscriminantPerGroup = Number of ANDs (default = 15).
%               options.cv = use a subset (10 percent) of training as validation set (default = 0).


% Hyper parameters
epsilon = 0.05;
momentum = 0.5;
maxepoch = 15;
nGroup = 24;
nDiscriminantPerGroup = 24;
cv = 0;
lambda = 0;
level = 1;
stage = 1;


has_name = false;
has_validation = false;

if nargin < 2
    error('At least two input arguments are required');
elseif nargin==2
    xtrain = varargin{1};
    ytrain = varargin{2};
elseif nargin==3
    xtrain = varargin{1};
    ytrain = varargin{2};    
    options = varargin{3};
    name = fieldnames(options);
    has_name = true;
elseif nargin==4
    xtrain = varargin{1};
    ytrain = varargin{2};    
    xvalid = varargin{3};
    yvalid = varargin{4};
    has_validation = true;
elseif nargin==5;
    xtrain = varargin{1};
    ytrain = varargin{2};    
    xvalid = varargin{3};
    yvalid = varargin{4};    
    options = varargin{5};
    name = fieldnames(options);
    has_name = true;
    has_validation = true;
elseif nargin > 5
    error('Too many input arguments');
end

if has_name
    for i = 1:length(name)
        switch name{i}
            case 'epsilon'
                epsilon = options.epsilon;
            case 'momentum'
                momentum = options.momentum;
            case 'maxepoch'
                maxepoch = options.maxepoch;
            case 'nGroup'
                nGroup = options.nGroup;
            case 'nDiscriminantPerGroup'
                nDiscriminantPerGroup = options.nDiscriminantPerGroup;
            case 'cv'
                cv = options.cv;
            case 'lambda'
                lambda = options.lambda;
            case 'level'
                level = options.level;
            case 'stage'
                stage = options.stage;
            otherwise
                warning(['Undefined option ' name{i} '...skipping']);
        end
    end
end

if level==0
    epsilon = 0.01;
    momentum = 0.5;
    if stage==1
        maxepoch = 15;
    else
        maxepoch = 6;
    end
    nGroup = 10;
    nDiscriminantPerGroup = 20;
end


if (has_validation && cv)
    cv = 0;
elseif cv
    fprintf('Using 10 percent of data for validation\n');
    npm = randperm(size(xtrain,2));
    nvalid = floor(.1*size(xtrain,2));
    xvalid = xtrain(:,npm(1:nvalid));
    yvalid = ytrain(:,npm(1:nvalid));
    xtrain = xtrain(:,npm(nvalid+1:end));
    ytrain = ytrain(:,npm(nvalid+1:end));
end
 

if lambda~=0 && has_validation
    warning('Both cross validation and regularization is active... turning off the regularizer');
    lambda = 0;
end

indexP = find(ytrain>0);
indexN = find(ytrain==0);



fprintf('Run clustering...');
ClusterDownsample = 10;
tic;
[~,cP] = kmeansML (nGroup,xtrain(:,indexP(1:ClusterDownsample:end)));
[~,cN]= kmeansML (nDiscriminantPerGroup,xtrain(:,indexN(1:ClusterDownsample:end)));
clTime = toc;
fprintf('Done. It took %f  \n',clTime);

n = size(xtrain,2);
fprintf('Number of training samples = %d \n',n);


discriminants = [];
centroids = [];
for p = 1:nGroup
    discriminants = [discriminants bsxfun(@minus,cP(:,p),cN)];
    centroids = [centroids 0.5*bsxfun(@plus,cP(:,p),cN)];
end;
discriminants = bsxfun(@rdivide,discriminants,sqrt(sum(discriminants.^2,1)));
discriminants = [discriminants;-sum(discriminants.*centroids,1)];

% discriminants = randn(size(xtrain,1)+1,nDiscriminantPerGroup*nGroup); %uncomment for random initialization

xtrain = [xtrain;ones(1,n)];
if has_validation
    xvalid = [xvalid;ones(1,length(yvalid))];
end

if level==0
 [discriminants totalerror] = UpdateDiscriminants_SB( single(xtrain), single(ytrain), single(discriminants),...
    maxepoch, nDiscriminantPerGroup, nGroup, single(epsilon), single(momentum));   
else
[discriminants totalerror] = UpdateDiscriminants( xtrain, ytrain, discriminants,...
    maxepoch, nDiscriminantPerGroup, nGroup, epsilon, momentum);
end
model.discriminants = discriminants;
model.totalerror = totalerror;
model.nGroup = nGroup;
model.nDiscriminantPerGroup = nDiscriminantPerGroup;

