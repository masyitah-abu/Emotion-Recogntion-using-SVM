clear;
clc;

DBPath = 'C:\Users\masyi\Desktop\Freelance\project\Syamizal(emotion recognition image)-150\FreelanceSyamizal\Databases\After Pre-Pocessing\4DBs (7 Expressions)'; % ubah ikot mana kamu simpan coding ni

%%  ************************ Read dataset *******************************
%%%%%% Training %%
Train_Angry_Folder =          fullfile(DBPath, '\JAFFE\Train\Angry');
Train_Disgusted_Folder =      fullfile(DBPath, '\JAFFE\Train\Disgusted');
Train_Fear_Folder =           fullfile(DBPath, '\JAFFE\Train\Fear');
Train_Happy_Folder =          fullfile(DBPath, '\JAFFE\Train\Happy');
Train_Neutral_Folder =        fullfile(DBPath, '\JAFFE\Train\Neutral');
%%%%%% Testing %%
Test_Angry_Folder =          fullfile(DBPath, '\JAFFE\Test\Angry');
Test_Disgusted_Folder =      fullfile(DBPath, '\JAFFE\Test\Disgusted');
Test_Fear_Folder =           fullfile(DBPath, '\JAFFE\Test\Fear');
Test_Happy_Folder =          fullfile(DBPath, '\JAFFE\Test\Happy');
Test_Neutral_Folder =        fullfile(DBPath, '\JAFFE\Test\Neutral');
database_N=1;
Cell_Size = 8;
%%%%%% Cell_size for LBP
if Cell_Size == 1
    CS=8;
CS2Excel=8;
elseif Cell_Size == 2
    CS=16;
CS2Excel=16;
elseif Cell_Size == 3
    CS=32;
CS2Excel=32;
elseif Cell_Size == 4
    CS=50;
CS2Excel=64;
elseif Cell_Size == 5
    CS=10;
CS2Excel=10;
elseif Cell_Size == 6
    CS=12;
CS2Excel=12;
elseif Cell_Size == 7
    CS=14;
CS2Excel=14;
elseif Cell_Size == 8
    CS=16;
CS2Excel=16;
elseif Cell_Size == 9
    CS=18;
CS2Excel=18;
elseif Cell_Size == 10
    CS=20;
CS2Excel=20;
elseif Cell_Size == 11
    CS=22;
CS2Excel=22;
elseif Cell_Size == 12
    CS=24;
CS2Excel=24;
elseif Cell_Size == 13
    CS=26;
CS2Excel=26;
elseif Cell_Size == 14
    CS=28;
CS2Excel=28;
elseif Cell_Size == 15
    CS=30;
CS2Excel=30;
elseif Cell_Size == 16
    CS=32;
CS2Excel=32;
elseif Cell_Size == 17
    CS=34;
CS2Excel=34;
elseif Cell_Size == 18
    CS=36;
CS2Excel=36;
elseif Cell_Size == 19
    CS=38;
CS2Excel=38;
elseif Cell_Size == 20
    CS=40;
CS2Excel=40;
elseif Cell_Size == 21
    CS=42;
CS2Excel=42;
elseif Cell_Size == 22
    CS=44;
CS2Excel=44;
elseif Cell_Size == 23
    CS=46;
CS2Excel=46;
elseif Cell_Size == 24
    CS=48;
CS2Excel=48;
elseif Cell_Size == 25
    CS=50;
CS2Excel=50;
elseif Cell_Size == 26
    CS=52;
CS2Excel=52;
elseif Cell_Size == 27
    CS=54;
CS2Excel=54;
elseif Cell_Size == 28
    CS=56;
CS2Excel=56;
elseif Cell_Size == 29
    CS=58;
CS2Excel=58;
elseif Cell_Size == 30
    CS=60;
CS2Excel=60;
elseif Cell_Size == 31
    CS=62;
CS2Excel=62;
elseif Cell_Size == 32
    CS=64;
CS2Excel=64;
end

%%  *********************** Feature Extraction for all data ******************************
[Train_Angry_Matrix,Train_No_Of_AN_Images] = LBP(CS,Train_Angry_Folder,'Loading Angry Expression...','Extract LBP Features for Training set');
[Train_Disgusted_Matrix,Train_No_Of_DI_Images] = LBP(CS,Train_Disgusted_Folder,'Loading Disgusted Expression...','Extract LBP Features for Training set');
[Train_Fear_Matrix,Train_No_Of_FE_Images] = LBP(CS,Train_Fear_Folder,'Loading Fear Expression...','Extract LBP Features for Training set');
[Train_Happy_Matrix,Train_No_Of_HA_Images] = LBP(CS,Train_Happy_Folder,'Loading Happy Expression...','Extract LBP Features for Training set');
[Train_neutral_Matrix,Train_No_Of_NE_Images] = LBP(CS,Train_Neutral_Folder,'Loading Neutral Expression...','Extract LBP Features for Training set');
[Test_Angry_Matrix,Test_No_Of_AN_Images] = LBP(CS,Test_Angry_Folder,'Loading Angry Expression...','Extract LBP Features for Test set');
[Test_Disgusted_Matrix,Test_No_Of_DI_Images] = LBP(CS,Test_Disgusted_Folder,'Loading Disgusted Expression...','Extract LBP Features for Test set');
[Test_Fear_Matrix,Test_No_Of_FE_Images] = LBP(CS,Test_Fear_Folder,'Loading Fear Expression...','Extract LBP Features for Test set');
[Test_Happy_Matrix,Test_No_Of_HA_Images] = LBP(CS,Test_Happy_Folder,'Loading Happy Expression...','Extract LBP Features for Test set');
[Test_neutral_Matrix,Test_No_Of_NE_Images] = LBP(CS,Test_Neutral_Folder,'Loading Neutral Expression...','Extract LBP Features for Test set');

Train_Matrix = vertcat(Train_Angry_Matrix, Train_Disgusted_Matrix, Train_Fear_Matrix, Train_Happy_Matrix, Train_neutral_Matrix);
Test_Matrix = vertcat(Test_Angry_Matrix, Test_Disgusted_Matrix, Test_Fear_Matrix, Test_Happy_Matrix, Test_neutral_Matrix);

%%  *********************** Divide into class in matrix *****
%  *********************** Training
Train_Class = zeros(size(length(Train_Matrix),1),1);
    Train_Class(1:  Train_No_Of_AN_Images,1) = 1;  % Angry;
    Train_Class(    Train_No_Of_AN_Images+1:...
                    Train_No_Of_AN_Images+Train_No_Of_DI_Images,1) = 2; %Disgusted;
    Train_Class(    Train_No_Of_AN_Images+Train_No_Of_DI_Images+1:...
                    Train_No_Of_AN_Images+Train_No_Of_DI_Images+Train_No_Of_FE_Images,1) = 3; %Afraid;
    Train_Class(    Train_No_Of_AN_Images+Train_No_Of_DI_Images+Train_No_Of_FE_Images+1:...
                    Train_No_Of_AN_Images+Train_No_Of_DI_Images+Train_No_Of_FE_Images+Train_No_Of_HA_Images,1) = 4; %Happy;
    Train_Class(    Train_No_Of_AN_Images+Train_No_Of_DI_Images+Train_No_Of_FE_Images+Train_No_Of_HA_Images+1:...
                    Train_No_Of_AN_Images+Train_No_Of_DI_Images+Train_No_Of_FE_Images+Train_No_Of_HA_Images+Train_No_Of_NE_Images,1) = 5; %neutral;
%  *********************** Testing
 Test_Class = zeros(size(length(Test_Matrix),1),1);
    Test_Class(1:  Test_No_Of_AN_Images,1) = 1;  % Angry;
    Test_Class(    Test_No_Of_AN_Images+1:...
                   Test_No_Of_AN_Images+Test_No_Of_DI_Images,1) = 2; %Disgusted;
    Test_Class(    Test_No_Of_AN_Images+Test_No_Of_DI_Images+1:...
                   Test_No_Of_AN_Images+Test_No_Of_DI_Images+Test_No_Of_FE_Images,1) = 3; %Afraid;
    Test_Class(    Test_No_Of_AN_Images+Test_No_Of_DI_Images+Test_No_Of_FE_Images+1:...
                   Test_No_Of_AN_Images+Test_No_Of_DI_Images+Test_No_Of_FE_Images+Test_No_Of_HA_Images,1) = 4; %Happy;
    Test_Class(    Test_No_Of_AN_Images+Test_No_Of_DI_Images+Test_No_Of_FE_Images+Test_No_Of_HA_Images+1:...
                   Test_No_Of_AN_Images+Test_No_Of_DI_Images+Test_No_Of_FE_Images+Test_No_Of_HA_Images+Test_No_Of_NE_Images,1) = 5; %neutral;
 %% *********************** Classification *****************
TestSet=Test_Matrix;
GroupTrain=Train_Class;
TrainingSet=Train_Matrix;
u=unique(GroupTrain);%1,2,3,4,5,6,7
numClasses=length(u);%7
t = templateEnsemble( 'GentleBoost' ,50, 'Tree' );
models = fitcecoc(TrainingSet,GroupTrain);
[MultiSVM_result]=predict(models,TestSet);

AN=0;DI=0;FE=0;HA=0;NE=0;
         
False_Expr=[];
Ind_Of_False_Expr=[];         
for i=1:length(Test_Class)
     if (MultiSVM_result(i)==Test_Class(i))
            if MultiSVM_result(i) == 1
            AN=AN+1;
            elseif MultiSVM_result(i) == 2
            DI=DI+1;
            elseif MultiSVM_result(i) == 3
            FE=FE+1;
            elseif MultiSVM_result(i) == 4
            HA=HA+1;
            elseif MultiSVM_result(i) == 5
            NE=NE+1;
            end
         Result(i)=1;
     else
         Result(i)=0;         
         False_Expr=[False_Expr;MultiSVM_result(i)];%  False_Expr(i)=knn_Result(i);
         Ind_Of_False_Expr=[Ind_Of_False_Expr;i];%  Ind_Of_False_Expr(i)=i;
     end
end
Accurancy=sum(Result)/length(Test_Class)*100;
Sheet_Na='LBP+SVM'; %Name of sheet
%%  ****************************** Confussion Matrix dia kira semua yg ada dlm folder JAFFE********************************
Percent='%';
fprintf('===================================================================\n');
fprintf('%s ||%s||%s||%s||%s\n', 'Expression', 'Train images', 'Test images', 'True Expressions', 'Accurancy');
fprintf('===========||============||===========||================||=========\n');
AN1=AN/Test_No_Of_AN_Images*100;
fprintf('Angry:     ||     %d     ||     %d     ||       %d        ||  %0.2f%s\n',Train_No_Of_AN_Images,Test_No_Of_AN_Images,AN,AN1,Percent);
fprintf('-----------||------------||-----------||----------------||---------\n');
DI1=DI/Test_No_Of_DI_Images*100;
fprintf('Disgusted: ||     %d     ||     %d     ||       %d        ||  %0.2f%s\n',Train_No_Of_DI_Images,Test_No_Of_DI_Images,DI,DI1,Percent);
fprintf('-----------||------------||-----------||----------------||---------\n');
FE1=FE/Test_No_Of_FE_Images*100;
fprintf('Fear:      ||     %d     ||     %d     ||       %d        ||  %0.2f%s\n',Train_No_Of_FE_Images,Test_No_Of_FE_Images,FE,FE1,Percent);
fprintf('-----------||------------||-----------||----------------||---------\n');
HA1=HA/Test_No_Of_HA_Images*100;
fprintf('Happy:     ||     %d     ||     %d     ||       %d        ||  %0.2f%s\n',Train_No_Of_HA_Images,Test_No_Of_HA_Images,HA,HA1,Percent);
fprintf('-----------||------------||-----------||----------------||---------\n');
NE1=NE/Test_No_Of_NE_Images*100;
fprintf('Neutral:   ||     %d     ||     %d     ||       %d        ||  %0.2f%s\n',Train_No_Of_NE_Images,Test_No_Of_NE_Images,NE,NE1,Percent);
fprintf('-----------||------------||-----------||----------------||---------\n');
fprintf('Total:     ||     %d    ||     %d    ||       %d       ||  %0.2f%s\n',length(Train_Class),length(Test_Class),sum(Result),Accurancy,Percent);
fprintf('===================================================================\n');
%========          
    for k=1:length(False_Expr)
            fprintf('Ind of false Expr %d=',Ind_Of_False_Expr(k));
            if False_Expr(k) == 1
            fprintf('Angry\n');
            elseif False_Expr(k) == 2
            fprintf('Disgusted\n');
            elseif False_Expr(k) == 3
            fprintf('Fear\n');
            elseif False_Expr(k) == 4
            fprintf('Happy\n');
            elseif False_Expr(k) == 5
            fprintf('Neutral\n');
            end
    end
%========          
%%  ***********************Each emotion & Overall accuracy***********************
Angry={Train_No_Of_AN_Images,Test_No_Of_AN_Images,AN,AN1};
Disgusted={Train_No_Of_DI_Images,Test_No_Of_DI_Images,DI,DI1};
Fear={Train_No_Of_FE_Images,Test_No_Of_FE_Images,FE,FE1};
Happy={Train_No_Of_HA_Images,Test_No_Of_HA_Images,HA,HA1};
Neutral={Train_No_Of_NE_Images,Test_No_Of_NE_Images,NE,NE1};
Total={length(Train_Class),length(Test_Class),sum(Result),Accurancy};
[Cmat,DA]= confusion_matrix(MultiSVM_result,Test_Class);
fprintf('Done\n');

%% Testing for one image%%
[filename, pathname] = uigetfile('*.tiff','Select the Testing Input FaceImage');
path = strcat(pathname,'\',filename);
Images = imread(path);
%%%%% noise %%
Images = imnoise(Images, 'salt & pepper' ,0.02);%higher the number(0.02) higher the noise the lower the percentage
%%%%% Pre-processing %%
FaceDetector = vision.CascadeObjectDetector();
BB = step(FaceDetector, Images);
Images = imcrop(Images,BB);
Images = imresize(Images,[160 160]); %% Image size for trained and testing must be the same because it is size sensitive
%%%%%%% Cropped Histogram Equalization Image
Images = histeq(Images); %%boleh add filter lain lg untuk naikkan percent
%%%%%% Feature Extraction %%%%
TestSet= extractLBPFeatures(Images,'CellSize',[CS CS]);
%[MultiSVM_result,score]= predict(models,TestSet);
MdlFinal = fitcensemble(TrainingSet,GroupTrain);
[label,score] = predict(MdlFinal,TestSet) %%percent tgk dekat score yg paling tinggi yg tu percentage 
%%%%  kalau masuk kat GUI untuk label dgn accuracy
if label == 1
    percentage=score(label);
    disp(percentage)
elseif label == 2
    percentage=score(label);
    disp(percentage)
elseif label == 3
    percentage=score(label);
    disp(percentage)
elseif label == 4
    percentage=score(label);
    disp(percentage)
elseif label == 5
    percentage=score(label);
    disp(percentage)
else
    disp('undefined')
end

