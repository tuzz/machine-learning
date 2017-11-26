function [J, grad] = lrCostFunction(theta, X, y, lambda)

% It turns out I jumped the gun and implemented the vectorized implementation of
% the logistic regression cost function last week. It would have been helpful to
% have all the guidance provided in this week's PDF, but it was satisfying to
% figure this out myself then see it give hints about using things like ':end'
% and using 'sum'.

addpath("../../week-3/ex2")
[J, grad] = costFunctionReg(theta, X, y, lambda);
rmpath("../../week-3/ex2")

end
