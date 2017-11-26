% This seems like a sensible thing to extract.
function [theta] = kthOneVsAll(X, y, k, lambda)
  % n is the number of features (the intercept has already been added to X).
  n = size(X, 2);

  % Specialise the cost function for the current value of k.
  costFunction = @(t)(lrCostFunction(t, X, (y == k), lambda));

  % Use Octave's built-in 'fmincg' function to minimise theta.
  initial_theta = zeros(n, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  theta = fmincg(costFunction, initial_theta, options);
end
