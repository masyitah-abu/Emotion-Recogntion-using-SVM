function [result] = multisvm(TrainingSet,GroupTrain,TestSet)
%Models a given training set with a corresponding group vector and 
%classifies a given test set using an SVM classifier according to a 
%one vs. all relation. 
%
%This code was written by Cody Neuburger cneuburg@fau.edu
%Florida Atlantic University, Florida USA
%This code was adapted and cleaned from Anand Mishra's multisvm function
%found at http://www.mathworks.com/matlabcentral/fileexchange/33170-multi-class-support-vector-machine/

u=unique(GroupTrain);
numClasses=length(u);
result = zeros(length(TestSet(:,1)),1);
waitbar3 = waitbar(0.1,'Multi-Class SVM classify','Name','Wait...'); %% WaitBar

%build models
models = fitcecoc(TrainingSet,GroupTrain);
%classify test cases
result=predict(models,TestSet);
close(waitbar3)%% WaitBar 