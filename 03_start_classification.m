load('rec_male_wu.mat')
load('rec_male_chunwang.mat')
load('rec_female_zhongjin.mat')
load('rec_female_kaixin.mat')

data1=rec_female_zhongjin(:,1:100);
numObservations = size(data1,2);
numObservationsTrain = floor(0.8*numObservations);
numObservationsValidation = numObservations - numObservationsTrain;
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:end);
tblTrain1 = data1(:,idxTrain);
tblValidation1 = data1(:,idxValidation);

data2=rec_female_kaixin(:,1:100);
numObservations = size(data2,2);
numObservationsTrain = floor(0.8*numObservations);
numObservationsValidation = numObservations - numObservationsTrain;
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:end);
tblTrain2 = data2(:,idxTrain);
tblValidation2 = data2(:,idxValidation);


data3=rec_male_wu(:,1:100);
numObservations = size(data3,2);
numObservationsTrain = floor(0.8*numObservations);
numObservationsValidation = numObservations - numObservationsTrain;
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:end);
tblTrain3 = data3(:,idxTrain);
tblValidation3 = data3(:,idxValidation);

data4=rec_female_zhongjin(:,1:100);
numObservations = size(data4,2);
numObservationsTrain = floor(0.8*numObservations);
numObservationsValidation = numObservations - numObservationsTrain;
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:end);
tblTrain4 = data4(:,idxTrain);
tblValidation4 = data4(:,idxValidation);

tblTrain = [tblTrain1 tblTrain2 tblTrain3 tblTrain4];
tblValidation = [tblValidation1 tblValidation2 tblValidation3 tblValidation4];

trainOutputs1 = (ones(size(tblTrain1,2),1));
valOutputs1 = (ones(size(tblValidation1,2),1));
trainOutputs2 = (zeros(size(tblTrain2,2),1));
valOutputs2 = (zeros(size(tblValidation2,2),1));
trainOutputs3 = (ones(size(tblTrain3,2),1));
valOutputs3 = (ones(size(tblValidation3,2),1));
trainOutputs4 = (ones(size(tblTrain4,2),1));
valOutputs4 = (ones(size(tblValidation4,2),1));

trainOutputs = [trainOutputs1 ; trainOutputs2 ;trainOutputs3 ;trainOutputs4];
valOutputs = [valOutputs1 ; valOutputs2 ; valOutputs3 ; valOutputs4];



% We can crate a table with the input data (1024x160) and add the class (1
% or 0) as the last column (similar to the example from the link)
trainTable = array2table(tblTrain');
trainData = [trainTable array2table(trainOutputs)];

labelName = "trainOutputs";
trainData = convertvars(trainData,labelName,'categorical');

valTable = array2table(tblValidation');
valData = [valTable array2table(valOutputs)];

labelName = "valOutputs";
valData = convertvars(valData,labelName,'categorical');



layers = [
featureInputLayer(1024)
fullyConnectedLayer(100)
reluLayer
fullyConnectedLayer(2)
softmaxLayer
classificationLayer
];


options = trainingOptions('sgdm', ...
'Shuffle','every-epoch', ...
'Plots','training-progress', ...
'Verbose',false);

net = trainNetwork(trainData,layers,options);

YPred = classify(net,valData);
labelName = "valOutputs";
YTest = valData{:,labelName};
accuracy = sum(YPred == YTest)/numel(YTest)

plotconfusion(YTest,YPred)

