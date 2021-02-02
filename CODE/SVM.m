clc;
clear;

%load images: row is feature, column is observation
X = MNIST_IMAGES('train-images.idx3-ubyte');
y = MNIST_LABELS('train-labels.idx1-ubyte');
XT = MNIST_IMAGES('t10k-images.idx3-ubyte');
yt = MNIST_LABELS('t10k-labels.idx1-ubyte');

%dimensionality reduction
DR = 2; %1: PCA, 2: LDA
if DR == 1 %PCA
    p=10;
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(X',  'NumComponents', p);
    display(p);
    W = COEFF(:, 1:p);
elseif DR == 2 %LDA
    W = LDA(X',y);
else
    disp('Invalid dimension reduction method selection');
    return;
end

%weight matrix
X = W' * X;
XT = W' * XT;

%train & test
kernel = 2; %libsvm, 0:linear, 1:polynomial, 2:rbf
numLabels = 10;
model = cell(numLabels,1);
disp('Start training...');
for i = 1:numLabels
    disp(i);
    if kernel == 0
        model{i} = svmtrain(double(y==(i-1)), X', '-s 0 -t 0 -b 1');
    elseif kernel == 1
        model{i} = svmtrain(double(y==(i-1)), X', '-s 0  -t 1 -b 1');
    elseif kernel == 2
        model{i} = svmtrain(double(y==(i-1)), X', '-s 0  -t 2 -b 1');
    else
        disp('Invalid Kernel Selection');
        return;
    end
end
disp('Start testing...');
prob = zeros(length(yt),numLabels);
for i = 1:numLabels
    disp(i);
    [~,~,p] = svmpredict(double(yt==(i-1)), XT', model{i}, '-b 1');
    prob(:,i) = p(:, model{i}.Label==1);
end
[~,pred] = max(prob,[],2);
accuracy = sum((pred-1) == yt) ./ length(yt);
disp(accuracy);


%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

function images = MNIST_IMAGES(filename)
%MNIST_IMAGES returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
images = double(images) / 255;

end

function labels = MNIST_LABELS(filename)
%MNIST_LABELS returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);

end

function [W] = LDA(X, y)
    numFeatures = size(X, 2);
    labels = unique(y);
    numLabels = length(labels);
    mu = zeros(numLabels, numFeatures);
    for i=1:numLabels
        mu(i,:) = mean( X((y==labels(i)),:) ); %row vector
    end
    mu_all = mean(X); 
    delta = 0.1;  
    SW = zeros(numFeatures,numFeatures);
    for i=1:numLabels
        S = cov(X);
        if(det(S)==0)
            S = S + delta * eye(numFeatures);
        end
        SW  = SW + S;
    end

    if(det(SW)==0)
        disp('Singular');
        pause;
    end

    SB = zeros(numFeatures,numFeatures);
    for i=1:numLabels
       Ni = sum( y==labels(i) );
       SB = SB + Ni * ( mu(i,:) - mu_all )' * ( mu(i,:) - mu_all );      
    end
    [W,~] = eigs(SB,SW, numLabels-1);
end