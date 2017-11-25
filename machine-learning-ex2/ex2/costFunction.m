function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Compute the cost function for logistic regression. We need to use the sigmoid
% function for hTheta so that the cost function is convex and will work with
% gradient descent (or more sophisticated minimisation algorithms.
hTheta = sigmoid(X * theta);

% Compute the y=1 and y=0 components of the cost function then combine them in
% a way that eliminates 'termForYIsOne' when y=0 and eliminates 'termForYIsZero'
% when y=1.
termForYIsOne = y .* log(hTheta);
termForYIsZero = (1 - y) .* log(1 - hTheta);
costs = -(termForYIsOne + termForYIsZero);

% The cost is the average cost for all training examples.
J = sum(costs) / m;

% Compute the gradient for each parameter of the model. Gradient descent uses
% this to determine which direction to step (i.e. whether to add/subtract from
% the parameter) in and how big a step it should take. We can reuse hTheta.
grad = X' * (hTheta - y) / m;

end
