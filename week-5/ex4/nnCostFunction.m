function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Include the bias unit in the input layer.
X = [ones(m, 1) X];

% Compute the activations of the hidden units.
hiddenActivations = sigmoid(X * Theta1');

% Include the bias unit in the hidden layer.
m2 = size(hiddenActivations, 1);
hiddenActivations = [ones(m2, 1) hiddenActivations];

% Compute the activations of the output units.
outputActivations = sigmoid(hiddenActivations * Theta2');

% Build a matrix of expected outputs for each training example.
expectedOutputs = [1:num_labels] == y;

% Apply the logistic regression cost function.
termForYIsOne = expectedOutputs .* log(outputActivations);
termForYIsZero = (1 - expectedOutputs) .* log(1 - outputActivations);
costs = -(termForYIsOne + termForYIsZero);

% Compute the unregularized cost.
% This is the sum of all output neuron costs for all training examples.
unregCost = sum(sum(costs)) / m;

% Exclude bias units from regularization.
Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

% Sum over all weights.
Theta1Sum = sum(sum(Theta1NoBias .^ 2));
Theta2Sum = sum(sum(Theta2NoBias .^ 2));

% Compute the cost of the regularization term.
regularizationCost = lambda / (2 * m) * (Theta1Sum + Theta2Sum);

% The total cost is the unregularized cost plus the regularization term.
J = unregCost + regularizationCost;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
