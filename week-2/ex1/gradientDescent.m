function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Initially, more confusion...
    %
    % I thought the cost function was to be used to calculate the derivative,
    % when actually it's not. There's no sum of squares going on here, we're
    % calculating the error for each training set example the same way, but then
    % taking the average of these errors as the derivative.
    %
    % The derivation of this partial derivative isn't shown in the videos. My
    % intuition for where this comes from is it's analagous to the derivative of
    % x^2 which is 2x, i.e. we don't need to do any squaring and we multiply by
    % a scalar of 2. This, presumably, is why we introduce a constant of 1/2
    % into the cost function to simplify the math.
    %
    % The derivative is the gradient of the line, so we need to move in the
    % opposite direction to it (hence the minus). We do so by an amount scaled
    % by the learning rate, which gradually moves us towards a minimum without
    % overshooting it.

    expected = y;
    actual = X * theta;
    errors = (actual - expected);
    derivative = X' * errors / m;
    theta -= alpha * derivative;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
