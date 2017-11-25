function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Initially I was confused and tried (theta' * X) before realising the 'x' in
% the hyptothesis equation refers to the variables of the regression line and
% not concrete training data.
%
% Instead, we need to calculate the result of applying the hypothesis (actual)
% and compare this against the (expected) values in 'y' to compute the cost
% using the square error function.
%
% This is the vectorised form of this computation as it doesn't use for-loops.
% This should be computationally more efficient.

expected = y;
actual = X * theta;
errors = (actual - expected);
squareErrors = errors .^ 2;
sumOfSquares = sum(squareErrors);
J = sumOfSquares / (2 * m);

% =========================================================================

end
