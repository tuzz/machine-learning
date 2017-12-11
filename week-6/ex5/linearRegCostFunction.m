function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples


% hypothesis vs. actual term:
expected = X * theta;
difference = expected - y;
squareErrors = difference .^ 2;
sumOfSquares = sum(squareErrors);
costWithoutReg = sumOfSquares / (2 * m);

% regularization term:
thetaNoBias = theta(2:end);
sumOfThetaSquares = sum(thetaNoBias .^ 2);
regCost = lambda / (2 * m) * sumOfThetaSquares;

J = costWithoutReg + regCost;

% gradient:
gradWithoutReg = (difference' * X) / m;
regGrad = lambda / m * theta;
regGrad(1) = 0; % don't regularize bias
grad = gradWithoutReg + regGrad';

grad = grad(:);

end
