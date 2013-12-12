clear
trainpath = './trainImages/'; %path to the training Images
labelpath = '../../trainLabels/'; %path to the training Labels
files = dir([labelpath '*.png']);
TTH = [0:.01:1];
for t = 1:length(TTH)
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    for i = 1:14
        mask = imread([labelpath files(i).name]);
        mask = double(mask>127);
        temp = load([ 'slice' num2str(i)]);
        bb = double(temp.clabels>TTH(t));

        TP = TP + sum(sum(bb(mask==1)));
        FP = FP + sum(sum(bb(mask==0)));
        FN = FN + sum(histc(bb(mask==1),0));
        TN = TN + sum(histc(bb(mask==0),0));

    end
   TNR(t) = TN/(TN+FP); 
PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval(t) = 2 * PR * RC/(PR + RC);
PRR(t) = PR;
RCC(t) = RC;
GMEAN(t) = sqrt(RCC(t) * TNR(t));
end



[fmax indmax] = max(Fval);
[gmax indgmax] = max(GMEAN);

labelpath = '../../testLabels/'; %path to the training Labels
files = dir([labelpath '*.png']);
TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i = 1:56
    mask = imread([labelpath files(i).name]);
    mask = double(mask>127);
    temp = load([ 'slice' num2str(i)]);
    bb = double(temp.clabels>TTH(indmax));
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    
end

PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval = 2 * PR * RC/(PR + RC)


TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i = 1:15
    mask = imread(testlabel,i);
    mask = 1-mask;
    mask = double(mask>0.5);
    temp = load([ 'testslice' num2str(i)]);
    bb = double(temp.clabels>TTH(indgmax));
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    TN = TN + sum(histc(bb(mask==0),0));
end

tNR = TN/(TN+FP);
RC = TP/(TP + FN);
gmean = sqrt(tNR*RC)

%==== 5th scale

clear
labelpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_label.tif';
trainpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_input.tif';
testpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_input.tif';
testlabel = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_label.tif';


offset = SquareNeighborhood(1);
load Centers_5scale   
Outdir = 'EMoutput15by15_5scaleNew_r2/';
TTH = [0:.01:1];
for t = 1:length(TTH)
    TP = 0;
    FP = 0;
    FN = 0;
    for i = 1:14
        temp = load([Outdir 'slice' num2str(i) '.mat']);
        bb = double(temp.clabels>TTH(t));
        
        mask = imread(labelpath,i);
        mask = 1-mask;
        temp = double(mask>0.5);
        temp = maxpooling(maxpooling(maxpooling(maxpooling(temp))));
        fvL = ConstructNeighborhoods(temp,offset,2);
        dist = TwoBlockDistance(fvL,centers);
        [~, ind] = min(dist,[],2);
        clabels = reshape(ind,size(temp)); 
        clabels(clabels>1) = 2;
        clabels = clabels - 1;
        mask = clabels;




        TP = TP + sum(sum(bb(mask==1)));
        FP = FP + sum(sum(bb(mask==0)));
        FN = FN + sum(histc(bb(mask==1),0));

    end

    PR = TP/(TP + FP);
    RC = TP/(TP + FN);
    Fval(t) = 2 * PR * RC/(PR + RC);
end

[fmax indm] = max(Fval);
TP = 0;
FP = 0;
FN = 0;
for i = 1:56
    temp = load([Outdir 'testslice' num2str(i) '.mat']);
    bb = double(temp.clabels>TTH(indm));
    
    mask = imread(testlabel,i);
    mask = 1-mask;
    temp = double(mask>0.5);
    temp = maxpooling(maxpooling(maxpooling(maxpooling(temp))));
    fvL = ConstructNeighborhoods(temp,offset,2);
    dist = TwoBlockDistance(fvL,centers);
    [~, ind] = min(dist,[],2);
    clabels = reshape(ind,size(temp)); 
    clabels(clabels>1) = 2;
    clabels = clabels - 1;
    mask = clabels;
    
    
    
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    
end

PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval = 2 * PR * RC/(PR + RC)


%==== 4th scale


clear
labelpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_label.tif';
trainpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_input.tif';
testpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_input.tif';
testlabel = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_label.tif';


offset = SquareNeighborhood(1);
load Centers_4scale   
Outdir = 'EMoutput15by15_4scaleNew/';
TTH = [0:.01:1];
for t = 1:length(TTH)
    TP = 0;
    FP = 0;
    FN = 0;
    for i = 1:14
        temp = load([Outdir 'slice' num2str(i) '.mat']);
        bb = double(temp.clabels>TTH(t));
        
        mask = imread(labelpath,i);
        mask = 1-mask;
        temp = double(mask>0.5);
        temp = maxpooling(maxpooling(maxpooling(temp)));
        fvL = ConstructNeighborhoods(temp,offset,2);
        dist = TwoBlockDistance(fvL,centers);
        [~, ind] = min(dist,[],2);
        clabels = reshape(ind,size(temp)); 
        clabels(clabels>1) = 2;
        clabels = clabels - 1;
        mask = clabels;




        TP = TP + sum(sum(bb(mask==1)));
        FP = FP + sum(sum(bb(mask==0)));
        FN = FN + sum(histc(bb(mask==1),0));

    end

    PR = TP/(TP + FP);
    RC = TP/(TP + FN);
    Fval(t) = 2 * PR * RC/(PR + RC);
end

[fmax indm] = max(Fval);
TP = 0;
FP = 0;
FN = 0;
for i = 1:56
    temp = load([Outdir 'testslice' num2str(i) '.mat']);
    bb = double(temp.clabels>TTH(indm));
    
    mask = imread(testlabel,i);
    mask = 1-mask;
    temp = double(mask>0.5);
    temp = maxpooling(maxpooling(maxpooling(temp)));
    fvL = ConstructNeighborhoods(temp,offset,2);
    dist = TwoBlockDistance(fvL,centers);
    [~, ind] = min(dist,[],2);
    clabels = reshape(ind,size(temp)); 
    clabels(clabels>1) = 2;
    clabels = clabels - 1;
    mask = clabels;
    
    
    
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    
end

PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval = 2 * PR * RC/(PR + RC)





%==== 3rd scale


clear
labelpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_label.tif';
trainpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_input.tif';
testpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_input.tif';
testlabel = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_label.tif';


offset = SquareNeighborhood(3);
load Centers_3scale   
Outdir = 'EMoutput15by15_3scaleNew_r2/';
TTH = [0:.01:1];
for t = 1:length(TTH)
    TP = 0;
    FP = 0;
    FN = 0;
    for i = 1:14
        temp = load([Outdir 'slice' num2str(i) '.mat']);
        bb = double(temp.clabels>TTH(t));
        
        mask = imread(labelpath,i);
        mask = 1-mask;
        temp = double(mask>0.5);
        temp = maxpooling(maxpooling(temp));
        fvL = ConstructNeighborhoods(temp,offset,2);
        dist = TwoBlockDistance(fvL,centers);
        [~, ind] = min(dist,[],2);
        clabels = reshape(ind,size(temp)); 
        clabels(clabels>1) = 2;
        clabels = clabels - 1;
        mask = clabels;




        TP = TP + sum(sum(bb(mask==1)));
        FP = FP + sum(sum(bb(mask==0)));
        FN = FN + sum(histc(bb(mask==1),0));

    end

    PR = TP/(TP + FP);
    RC = TP/(TP + FN);
    Fval(t) = 2 * PR * RC/(PR + RC);
end

[fmax indm] = max(Fval);
TP = 0;
FP = 0;
FN = 0;
for i = 1:56
    temp = load([Outdir 'testslice' num2str(i) '.mat']);
    bb = double(temp.clabels>TTH(indm));
    
    mask = imread(testlabel,i);
    mask = 1-mask;
    temp = double(mask>0.5);
    temp = maxpooling(maxpooling(temp));
    fvL = ConstructNeighborhoods(temp,offset,2);
    dist = TwoBlockDistance(fvL,centers);
    [~, ind] = min(dist,[],2);
    clabels = reshape(ind,size(temp)); 
    clabels(clabels>1) = 2;
    clabels = clabels - 1;
    mask = clabels;
    
    
    
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    
end

PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval = 2 * PR * RC/(PR + RC)


%==== 2nd scale


clear
labelpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_label.tif';
trainpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/train_input.tif';
testpath = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_input.tif';
testlabel = '/usr/sci/crcnsdata/ellismanData/amira_markups/learning/multi_radon_context/PNGFILES/test_label.tif';


offset = SquareNeighborhood(5);
load Centers_2scale   
Outdir = 'EMoutput15by15_2scaleNew_r2/';
TTH = [0:.01:1];
for t = 1:length(TTH)
    TP = 0;
    FP = 0;
    FN = 0;
    for i = 1:14
        temp = load([Outdir 'slice' num2str(i) '.mat']);
        bb = double(temp.clabels>TTH(t));
        
        mask = imread(labelpath,i);
        mask = 1-mask;
        temp = double(mask>0.5);
        temp = maxpooling(temp);
        fvL = ConstructNeighborhoods(temp,offset,2);
        dist = TwoBlockDistance(fvL,centers);
        [~, ind] = min(dist,[],2);
        clabels = reshape(ind,size(temp)); 
        clabels(clabels>1) = 2;
        clabels = clabels - 1;
        mask = clabels;




        TP = TP + sum(sum(bb(mask==1)));
        FP = FP + sum(sum(bb(mask==0)));
        FN = FN + sum(histc(bb(mask==1),0));

    end

    PR = TP/(TP + FP);
    RC = TP/(TP + FN);
    Fval(t) = 2 * PR * RC/(PR + RC);
end

[fmax indm] = max(Fval);
TP = 0;
FP = 0;
FN = 0;
for i = 1:56
    temp = load([Outdir 'testslice' num2str(i) '.mat']);
    bb = double(temp.clabels>TTH(indm));
    
    mask = imread(testlabel,i);
    mask = 1-mask;
    temp = double(mask>0.5);
    temp = maxpooling(temp);
    fvL = ConstructNeighborhoods(temp,offset,2);
    dist = TwoBlockDistance(fvL,centers);
    [~, ind] = min(dist,[],2);
    clabels = reshape(ind,size(temp)); 
    clabels(clabels>1) = 2;
    clabels = clabels - 1;
    mask = clabels;
    
    
    
    
    TP = TP + sum(sum(bb(mask==1)));
    FP = FP + sum(sum(bb(mask==0)));
    FN = FN + sum(histc(bb(mask==1),0));
    
end

PR = TP/(TP + FP);
RC = TP/(TP + FN);
Fval = 2 * PR * RC/(PR + RC)
