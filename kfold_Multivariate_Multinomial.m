function error = kfold_Multivariate_Multinomial(labels_train, features_train, labels_test, features_test) 

K = 10; %10-fold cross validation
n = length(labels_train)/K;
%Prior and likelihood matrix 
prior = zeros(1,26);
likelihood = zeros(26,16);

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
        for l = 1:16
            for k = 0:15
                likelihood(j,l,k+1) = (1 + sum(features(:,l) == k))/(size(features,1) + size(features,2));
            end
        end      
    end
    
    for j = 1:length(labels_validation)
        for k = 1:26
            temp = 0;
            for l = 1:16
                temp = log(likelihood(k,l,features_validation(j,l)+1))*(features_validation(j,l));
                if(isnan(temp))
                    temp = 0;
                end
                check(k) = check(k) + temp;    
            end 
            check(k) = log(prior(k)) + check(k);
        end
        max_index = find(check == max(check));
        prediction(j) = 'A' -1 + max_index;
        check(:) = 0;
    end
    
    predictions = char(prediction)';
    accuracy(i) = sum(predictions == labels_validation)/length(labels_validation);
   
end
    
error = 1 - mean(accuracy);

end
