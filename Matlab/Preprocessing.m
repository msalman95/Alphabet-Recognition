function [labels_train,features_train,labels_test,features_test] = Preprocessing(file)

features = file.data;
labels = cell2mat(file.textdata);
labels_train = [];
features_train = [];
labels_test = [];
features_test = [];

for i = 1:26
    %Seperate the features and label for each class
    labels_class = labels(labels == 'A' - 1 + i);
    features_class = features(labels == 'A' - 1 + i,:);
    index = randperm(length(labels_class));
    labels_train = [labels_train; labels_class(index(1:650))];
    features_train = [features_train; features_class(index(1:650),:)];
    labels_test = [labels_test; labels_class(index(651:end))];
    features_test = [features_test; features_class(index(651:end),:)];
end

%Shuffle the training data 
shuffle = randperm(length(labels_train));
labels_train = labels_train(shuffle);
features_train = features_train(shuffle,:);
%Shuffle the test data
shuffle = randperm(length(labels_test));
features_test = features_test(shuffle,:);
labels_test = labels_test(shuffle);

end
