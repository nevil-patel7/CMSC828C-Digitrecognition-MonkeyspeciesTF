% load the images
[training_images,training_labels] = TRAIN_IMAGES(); 
[testing_images,testing_labels] = TEST_IMAGES(); 
net = alexnet;
CLASS = 10;
freeze = 1; %For Conv Layer param freeze = 1, Unfreeze = 0

%replace the NN_layers
NN_layers = net.Layers;
NN_layers(23) = fullyConnectedLayer(CLASS,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
NN_layers(25) = classificationLayer;

%Freeze Convolution Layer param 
if freeze == 1
    for i = 1:numel(NN_layers)-3 %for all the previous NN_layers
    if isprop(NN_layers(i),'WeightLearnRateFactor')
        NN_layers(i).WeightLearnRateFactor = 0;
    end
    if isprop(NN_layers(i),'WeightL2Factor')
        NN_layers(i).WeightL2Factor = 0;
    end
    if isprop(NN_layers(i),'BiasLearnRateFactor')
        NN_layers(i).BiasLearnRateFactor = 0;
    end
    if isprop(NN_layers(i),'BiasL2Factor')
        NN_layers(i).BiasL2Factor = 0;
    end
end
end


%training param
param = trainingOptions('sgdm','MiniBatchSize',64,'MaxEpoch',2,... %number of epoch
    'InitialLearnRate',0.0001,'Plots','training-progress');

% train the network
train_nn = trainNetwork(training_images,training_labels,NN_layers,param); 

% test the network
[predicted_labels,score] = classify(train_nn,testing_images); 
accuracy = mean(predicted_labels == testing_labels'); %calculate the accuracy
display(accuracy)


function [training_images,training_labels] = TRAIN_IMAGES()
    training_images = [];
    training_labels = [];
    for i = 1:10 
        D = sprintf('training/n%d',i-1); 
        S = dir(fullfile(D,'*.jpg')); 
        for k = 1:numel(S) 
            F = fullfile(D,S(k).name); 
            I = imread(F);
            training_images(:,:,:,end+1) = imresize(I,[227 227]); 
            training_labels(:,end+1) = i; 
        end
    end
    training_images = training_images(:,:,:,(2:end)); 
    training_labels = categorical(training_labels); 
end

function [testing_images,testing_labels] = TEST_IMAGES()
    testing_images = [];
    testing_labels = [];
    for i = 1:10 
        D = sprintf('validation/n%d',i-1);
        S = dir(fullfile(D,'*.jpg')); 
        for k = 1:numel(S) 
            F = fullfile(D,S(k).name); 
            I = imread(F);
            testing_images(:,:,:,end+1) = imresize(I,[227 227]); 
            testing_labels(:,end+1) = i; 
        end
    end
    testing_images = testing_images(:,:,:,(2:end)); 
    testing_labels = categorical(testing_labels);
end
