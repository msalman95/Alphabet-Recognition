function error = kfold_Gaussian(labels_train, features_train, labels_test, features_test) 

K = 10; %10-fold cross validation
n = length(labels_train)/K;
%Prior and likelihood matrix 
prior = zeros(1,26);

%Classification using Multinomial Naive Bayes
check = zeros(1,26);
accuracy = zeros(1,K);

for i = 1:K
    labels_validation = labels_train(((i-1)*n+1:1:i*n));
    features_validation = features_train(((i-1)*n+1:1:i*n),:);
    labels_kfold = [labels_train(1:(i-1)*n); labels_train(i*n+1:end)];
    features_kfold = [features_train(1:(i-1)*n,:); features_train(i*n+1:end,:)];
    for j = 1:26
        prior(j) = sum(labels_kfold == 'A' - 1 + j)/length(labels_kfold);
        features = features_kfold(labels_kfold == 'A' - 1 + j,:);
        Meu(j,:) = mean(features);
        sigma(j,:) = sqrt(var(features));
    end
    
    for j = 1:length(labels_validation)
        for k = 1:26
            check(k) =  prior(k)*prod((exp((-1/2)*(((features_validation(j,:) - Meu(k,:)).^2)./(sigma(k,:).*sigma(k,:)))))./(sqrt(2*pi.*sigma(k,:))));
        end
        max_index = find(check == max(check));
        prediction(j) = 'A' -1 + max_index;
    end
    
    predictions = char(prediction)';
    accuracy(i) = sum(predictions == labels_validation)/length(labels_validation);
   
end
    
error = 1 - mean(accuracy);

end
