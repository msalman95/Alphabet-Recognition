function [weights_final bias_final] = Logistic_Regression(features_train, labels_train)

%Divide into training set and validation set
labels_training = labels_train(1:15000);
labels_validation = labels_train(15001:end);
features_training = features_train(1:15000,:);
features_validation = features_train(15001:end,:);

%Use Xaviers initialization for the weights and bias
bias = -sqrt(6/(16 + 26)) + 2*sqrt(6/(16 + 26))*rand(26,1);
w = -sqrt(6/(16 + 26)) + 2*sqrt(6/(16 + 26))*rand(26,16);
%Initialize Variables
epoch = 50;
Entropy = zeros(epoch,1);
Accuracy = zeros(epoch,1);
learning_rate = 0.05;
alpha = 0.35;
delta_bias = zeros(size(bias));
delta_w = zeros(size(w));

%Temporary variables
labels = zeros(26,1);
min_error = Inf;

%Stopping Condition
count = 0;

for i = 1:epoch
    %Shuffle the data for SGD
    shuffle = randperm(length(labels_training));
 
    %Pass over an epoch
    for l = 1:length(labels_training)
        class = labels_training(shuffle(l));
        labels(class - 'A' + 1) = 1;
        
        %Forward Propogation
        total = w*features_training(shuffle(l),:)' + bias;
        output = softmax(total);
        
        %Backpropogation to update weights with momentum
        delta_bias = alpha*delta_bias + learning_rate*(output - labels);
        delta_w = alpha*delta_w + learning_rate*(output - labels)*features_training(shuffle(l),:);
        bias = bias - delta_bias;
        w = w - delta_w;
        
        labels(:) = 0;
    end
    
    accuracy = 0;
    %Test the trained network on validation set and calculate the error
    for l = 1:length(labels_validation)
        class = labels_validation(l);
        labels(class - 'A' + 1) = 1;
        total = w*features_validation(l,:)' + bias;
        output = softmax(total);
        
        max_index = find(output == max(output));
        prediction(l) = char('A' + max_index - 1);
        %Calculate the accuracy
        if (labels_validation(l) == prediction(l))
            accuracy = accuracy + 1;
        end
        
        %Calculate the entropy error
        error = -labels.*log(output);
        error(isnan(error)) = 0;
        Entropy(i) = Entropy(i) + sum(error);
        labels(:) = 0;
    end
    Accuracy(i) = accuracy*100/length(labels_validation);
    Entropy(i) = Entropy(i)/length(labels_validation);
       
    fprintf('The cross entropy error on the validation set is: %.3f, Epoch Number: %d\n',Entropy(i),i);
    fprintf('The accuracy on the validation set is: %.3f%%, Epoch Number: %d\n',Accuracy(i),i);
    %Early Stopping
    if (Entropy(i) < min_error)
        min_error = Entropy(i);
        weights_final = w;
        bias_final = bias;
        count = 0;
    else
        count = count + 1;
    end
    
    if (count == 3)
        break;
    end
    %Initialize the temporary variables back to zero
    labels(:) = 0;
    delta_bias(:,:) = 0;
    delta_w(:,:) = 0;
    
end
    
end
        




