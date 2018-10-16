function [w1_final b1_final w2_final b2_final] = OneHiddenLayer_Training(features_train, labels_train)

%Divide into training set and validation set
labels_training = labels_train(1:15000);
labels_validation = labels_train(15001:end);
features_training = features_train(1:15000,:);
features_validation = features_train(15001:end,:);

%Initialize the variables
epoch = 100;
Entropy = zeros(epoch,1);
Accuracy = zeros(epoch,1);
learning_rate = 0.1;
alpha = 0.2;
hidden_neurons = 30;
%Use Xaviers initialization for the weights and bias
w_layer1 = -sqrt(6/(16 + hidden_neurons)) + 2*sqrt(6/(16 + hidden_neurons))*rand(hidden_neurons,16);
b_layer1 = -sqrt(6/(16 + hidden_neurons)) + 2*sqrt(6/(16 + hidden_neurons))*rand(hidden_neurons,1);
w_layer2 = -sqrt(6/(26 + hidden_neurons)) + 2*sqrt(6/(26 + hidden_neurons))*rand(26,hidden_neurons);
b_layer2 = -sqrt(6/(26 + hidden_neurons)) + 2*sqrt(6/(26 + hidden_neurons))*rand(26,1);


%Temporary variables
labels = zeros(26,1);
min_error = Inf;
delta_b1 = zeros(size(b_layer1));
delta_w1 = zeros(size(w_layer1));
delta_b2 = zeros(size(b_layer2));
delta_w2 = zeros(size(w_layer2));


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
        v1 = w_layer1*features_training(shuffle(l),:)' + b_layer1;
        o1 = 1./(1 + exp(-v1));
        total = w_layer2*o1 + b_layer2;
        output = softmax(total);
        
        %Backpropogation to update weights with momentum
        delta_b2 = alpha*delta_b2 + learning_rate*(output - labels);
        delta_w2 = alpha*delta_w2 + learning_rate*(output - labels)*o1';
        delta_b1 = alpha*delta_b1 + (learning_rate*(output - labels)'*w_layer2)'.*o1.*(1-o1);
        delta_w1 = alpha*delta_w1 + ((learning_rate*(output - labels)'*w_layer2)'.*o1.*(1-o1))*features_training(shuffle(l),:);
        
        b_layer2 = b_layer2 - delta_b2;
        w_layer2 = w_layer2 - delta_w2;
        b_layer1 = b_layer1 - delta_b1;
        w_layer1 = w_layer1 - delta_w1;
        
        labels(:) = 0;
    end
    
    accuracy = 0;
    %Test the trained network on validation set and calculate the error
    for l = 1:length(labels_validation)
        class = labels_validation(l);
        labels(class - 'A' + 1) = 1;
        
        %Forward Propogation
        v1 = w_layer1*features_validation(l,:)' + b_layer1;
        o1 = 1./(1 + exp(-v1));
        total = w_layer2*o1 + b_layer2;
        output = softmax(total);
        
        max_index = find(output == max(output));
        prediction = char('A' + max_index - 1);
        %Calculate the accuracy
        if (labels_validation(l) == prediction)
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
       
    fprintf('The cross validation error is: %.3f, Epoch Number: %d\n',Entropy(i),i);
    fprintf('The average accuracy after cross validation is: %.2f%%\n',Accuracy(i));
    %Early Stopping
    if (Entropy(i) < min_error)
        min_error = Entropy(i);
        w1_final = w_layer1;
        b1_final = b_layer1;
        w2_final = w_layer2;
        b2_final = b_layer2;
        count = 0;
    else
        count = count + 1;
    end
    
    if (count == 10)
        break;
    end
    %Initialize the temporary variables back to zero
    labels(:) = 0;
    delta_b1(:,:) = 0;
    delta_w1(:,:) = 0;
    delta_b2(:,:) = 0;
    delta_w2(:,:) = 0;
    
end
    
end
        




