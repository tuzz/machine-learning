function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculate the scalar of the regularized cost.
scalar = lambda / (2 * m);

% Extract all parameters except the intercept.
thetaNoIntercept = theta(2:end);

% Calculate the cost incurred by regularization.
regularizeCost = scalar * sum(thetaNoIntercept .^ 2);

% Use the existing cost function to compute cost and gradient.
[unregCost, unregGrad] = costFunction(theta, X, y);

% The cost is the existing cost function plus the regularize cost.
J = unregCost + regularizeCost;

% Calculate the scalar of the regularized gradient.
scalar = lambda / m;

% Calculate the gradient of the cost incurred by regularization.
regularizeGrad = scalar * theta;

% Don't regularize the gradient of the intercept parameter.
regularizeGrad(1) = 0;

% The gradient is the existing gradient plus the regularize gradient.
grad = unregGrad + regularizeGrad;

% =============================================================

end
