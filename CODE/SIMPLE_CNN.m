% load the dataset
[images,labels] = TRAIN_IMAGES(); 
[test_images,test_labels] = TEST_IMAGES(); 

%bulid the network
NN_layers = [
    imageInputLayer([150 150 3]) 
    convolution2dLayer(10,6,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,16,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,32,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,32,'Padding','same')
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(10) 
    softmaxLayer 
    classificationLayer]; 

%traning param
param = trainingOptions('sgdm', 'InitialLearnRate',0.01, 'MaxEpochs',4,...% EPOCHS
    'Shuffle','every-epoch', 'Plots','training-progress'); 

%train the net
net = trainNetwork(images,labels,NN_layers,param);

%test the trained network
[predicted_labels,score] = classify(net,test_images); %test the neteork on the testing images
accuracy = mean(predicted_labels == test_labels'); %calculate the accuracy
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
            training_images(:,:,:,end+1) = imresize(I,[150 150]); 
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
            testing_images(:,:,:,end+1) = imresize(I,[150 150]); 
            testing_labels(:,end+1) = i; 
        end
    end
    testing_images = testing_images(:,:,:,(2:end)); 
    testing_labels = categorical(testing_labels);
end