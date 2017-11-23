function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find the indices of positive and negative training examples.
positive_indices = find(y == 1);
negative_indices = find(y == 0);

% Fetch the scores for the positive and negative indices.
positive_scores = X(positive_indices, [1 2]);
negative_scores = X(negative_indices, [1 2]);

% Plot '+' for positives and 'o' for negatives.
plot(positive_scores(:, 1), positive_scores(:, 2), 'k+', 'MarkerSize', 7, 'LineWidth', 2);
plot(negative_scores(:, 1), negative_scores(:, 2), 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'y');

% =========================================================================

hold off;

end
