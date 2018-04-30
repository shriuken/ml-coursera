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

%% cost function J(theta) = (1/2m) sum_i=1_m(h_theta(x^i)-y(i)^2)

%% given X (features) Nx2
%% y (actual results) Nx1
%% heta (parameters)  2x1AQ

h_theta = X * theta;
J = (1 / 2 / m) * sum((h_theta-y).^2);

% =========================================================================

end
