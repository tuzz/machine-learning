function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% When the hypothesis is applied, it returns the probability that y=1,
% i.e. P(y=1|x;theta). Therefore, we can 'predict' whether the student is
% admitted by checking if there's more than a 50% probability that y=1.
hTheta = sigmoid(X * theta);
p = hTheta >= 0.5;

end
