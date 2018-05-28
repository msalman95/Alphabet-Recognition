%**************************************************************************
clc;clear;
%Get the whole data in struct format
file = importdata('letter-recognition.data');

%Call the Preprocessing function to create test and training data
[labels_train,features_train,labels_test,features_test] = Preprocessing(file);

%Normalize the features to range of [0 1]
features_train = features_train./15;
features_test = features_test./15;

%%
%Finding the prior probability of each class 
prior = zeros(1,26);

for i = 1:26
    prior(i) = sum(labels_train == 'A' - 1 + i)/length(labels_train);
end

%Seperate the features of each class and find the probability matrix(each
%feature)
likelihood = zeros(26,16);
for i = 1:26
    features = features_train(labels_train == 'A' - 1 + i,:);
    likelihood(i,:) = (1 + sum(features))./(sum(sum(features)) + 16);
end

%Find the 10-fold cross validation error
error = kfold_Multinomial(labels_train, features_train, labels_test, features_test);
fprintf('The cross validation error for Multinomial Naive Bayes: %.2f\n',error);

%Classification using Multinomial Naive Bayes
check = zeros(1,26);
prediction = zeros(1,length(labels_test));
for i = 1:length(labels_test)
    for j = 1:26
        check(j) = log(prior(j)) + sum(log(likelihood(j,:)).*features_test(i,:));
    end
    max_index = find(check == max(check));
    prediction(i) = 'A' -1 + max_index;
end

%Find the Accuracy for Multinomial Naive Bayes
predictions = char(prediction)';
accuracy = sum(predictions == labels_test)*100/length(labels_test);
fprintf('The accuracy for Multinomial Naive Bayes: %.2f%%\n',accuracy);

%Calculate the Confusion Matrix 
Confusion_Matrix = confusionmat(labels_test,predictions);

%**************************************************************************
%***********************Gaussian Naive Bayes*******************************
%**************************************************************************
%%
%Implement Gaussian Naive Bayes
Meu = zeros(26,16);             %Calculate the mean for each class and feature
sigma = zeros(26,16);           %Calculate the std for each class and feature

for i = 1:26
    features = features_train(labels_train == 'A' - 1 + i,:);
    Meu(i,:) = mean(features);
    sigma(i,:) = sqrt(var(features));
end

%Classify using the trained Gaussian Naive Bayes
check_gauss = zeros(1,26);
for i = 1:length(labels_test)
    for j = 1:26
        check_gauss(j) = prior(j)*prod((exp((-1/2)*(((features_test(i,:) - Meu(j,:)).^2)./(sigma(j,:).*sigma(j,:)))))./(sqrt(2*pi.*sigma(j,:))));
    end
    max_index = find(check_gauss == max(check_gauss));
    prediction(i) = 'A' -1 + max_index;
end

%Find the 10-fold cross validation error
error = kfold_Gaussian(labels_train, features_train, labels_test, features_test);
fprintf('The cross validation error for Gaussian Naive Bayes: %.2f\n',error);

%Find the Accuracy for Gaussian Naive Bayes
predictions_gauss = char(prediction)';
accuracy = sum(predictions_gauss == labels_test)*100/length(labels_test);
fprintf('The accuracy for Gaussian Naive Bayes: %.2f%%\n',accuracy);

%Calculate the Confusion Matrix 
Confusion_MatrixG = confusionmat(labels_test,predictions_gauss);

%**************************************************************************
%*******************Multivariate Multinomial Naive Bayes*******************
%**************************************************************************
%%
%Implement Multivariate Multinomial Naive Bayes with smoothing
likelihood_mul = zeros(26,16,16);
for i = 1:26
    features = features_train(labels_train == 'A' - 1 + i,:);
    for j = 1:16
        for k = 0:15
            likelihood_mul(i,j,k+1) = (1 + sum(features(:,j) == k))/(size(features,1) + size(features,2));
        end
    end
end

%Find the 10-fold cross validation error
error = kfold_Multivariate_Multinomial(labels_train, features_train, labels_test, features_test);
fprintf('The cross validation error for Multivariate Multinomial Naive Bayes: %.2f\n',error);

%Classification using Multivariate Multinomial Naive Bayes
check_mul = zeros(1,26);
temp = 0;
prediction = zeros(1,length(labels_test));
for i = 1:length(labels_test)
    for j = 1:26
        temp = 0;
        for k = 1:16
            temp = log(likelihood_mul(j,k,features_test(i,k)+1))*(features_test(i,k));
            if(isnan(temp))
                temp = 0;
            end
            check_mul(j) = check_mul(j) + temp;
        end
        check_mul(j) = log(prior(j)) + check_mul(j);
    end
    max_index = find(check_mul == max(check_mul));
    prediction(i) = 'A' -1 + max_index;
    check_mul(:) = 0;
end

%Find the Accuracy for Gaussian Naive Bayes
predictions_mul = char(prediction)';
accuracy = sum(predictions_mul == labels_test)*100/length(labels_test);
fprintf('The accuracy for Multivariate Multinomial Naive Bayes: %.2f%%\n',accuracy);
%Calculate the Confusion Matrix 
Confusion_MatrixM = confusionmat(labels_test,predictions_mul);

%%
[w, bias] = Logistic_Regression(features_train, labels_train);
fprintf('*****************************************************************\n');
fprintf('*************************End of training*************************\n');
fprintf('*****************************************************************\n\n');

o = zeros(1,26);
accuracy = 0;
prediction = zeros(size(labels_test));

for i = 1:length(labels_test)
    total = w*features_test(i,:)' + bias; 
    output = softmax(total);
    
    max_index = find(output == max(output));
    prediction(i) = char('A' + max_index - 1);
    
    if (labels_test(i) == prediction(i))
        accuracy = accuracy + 1;
    end
end

acc_op = (accuracy*100)/length(labels_test);
fprintf('The accuracy for Logistic Regression: %.2f%%\n',acc_op);
%Calculate the confusion matrix
Confusion_Matrix = confusionmat(labels_test,char(prediction));

%**************************************************************************
%********************ONE HIDDEN LAYER**************************************
%**************************************************************************
%%
tic;
[w_layer1 b_layer1 w_layer2 b_layer2] = OneHiddenLayer_Training(features_train, labels_train);
fprintf('*****************************************************************\n');
fprintf('*************************End of training*************************\n');
fprintf('*****************************************************************\n\n');

o = zeros(1,26);
accuracy = 0;
prediction = zeros(size(labels_test));

for i = 1:length(labels_test)
    %Forward Propogation
    v1 = w_layer1*features_test(i,:)' + b_layer1;
    o1 = 1./(1 + exp(-v1));
    total = w_layer2*o1 + b_layer2;
    output = softmax(total);
 
    max_index = find(output == max(output));
    prediction(i) = char('A' + max_index - 1);
    
    if (labels_test(i) == prediction(i))
        accuracy = accuracy + 1;
    end
end

acc_op = (accuracy*100)/length(labels_test);
fprintf('The accuracy with one hidden layer: %.2f%%\n',acc_op);
%Calculate the confusion matrix
Confusion_Matrix = confusionmat(labels_test,char(prediction));
toc;

%**************************************************************************
%********************TWO HIDDEN LAYER**************************************
%**************************************************************************
%%
[w_layer1 b_layer1 w_layer2 b_layer2 w_layer3 b_layer3] = TwoHiddenLayer_Training(features_train, labels_train);
fprintf('*****************************************************************\n');
fprintf('*************************End of training*************************\n');
fprintf('*****************************************************************\n\n');

o = zeros(1,26);
accuracy = 0;
prediction = zeros(size(labels_test));

for i = 1:length(labels_test)
    
     %Forward Propogation
     v1 = w_layer1*features_test(i,:)' + b_layer1;
     o1 = 1./(1 + exp(-v1));
     v2 = w_layer2*o1 + b_layer2;
     o2 = 1./(1 + exp(-v2));
     total = w_layer3*o2 + b_layer3;
     output = softmax(total);
     
     max_index = find(output == max(output));
     prediction(i) = char('A' + max_index - 1);
     %Calculate the accuracy
     if (labels_test(i) == prediction(i))
         accuracy = accuracy + 1;
     end

end

acc_op = (accuracy*100)/length(labels_test);
fprintf('The accuracy with two hidden layer: %.2f%%\n',acc_op);
%Calculate the confusion matrix
Confusion_Matrix = confusionmat(labels_test,char(prediction));