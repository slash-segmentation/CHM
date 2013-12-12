function [outputB outputF] = EvaluateAndOrNetMX (x, model)


discriminants = model.discriminants;
nGroup = model.nGroup;
nDiscriminantPerGroup = model.nDiscriminantPerGroup;
n = size(x,2);
% x = [x; ones(1,n)];
if isa(discriminants,'single')
    outputF = genOutput_SB(single(x), discriminants, nGroup, nDiscriminantPerGroup);
    outputF = double(outputF);
else
    x = [x; ones(1,n)];
    outputF = genOutput(x, discriminants, nGroup, nDiscriminantPerGroup);
end

outputB = double(outputF>0.5);
