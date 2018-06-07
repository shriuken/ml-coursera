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

% J(theta) = (1 / m) *
%               sum(i=1, m)[sum(k=1, K)[-y_k^(i) * log((h_theta(x^i))_k) - (1-y_k^i) * log(1 - (h_theta(x^i))_k)]]

a1 = [ones(size(X, 1), 1) X]; % 5000 x 401
z2 = a1 * Theta1'; % 5000 x 25 
a2 = [ones(m, 1) sigmoid(z2)]; % 5000 x 26
z3 = a2 * Theta2'; % 5000 x 10
h_theta = sigmoid(z3); % 5000 x 10

% need to turn y into a 5000 x 10 where the initial value of y becomes that column's 1 value.
y_matrix = zeros(size(y, 1), num_labels);
for row=1:1:m
  y_matrix(row, y(row)) = 1;
end

J = (1 / m) * sum(sum( -y_matrix .* log((h_theta)) - (1 .- y_matrix) .* log(1 .- h_theta)));
regConstant = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
% Adding regularization term
J = J + regConstant;

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

% size(Theta1_grad) % 25 x 401
% size(Theta2_grad) % 10 x 26
% size(X) % 5000 x 400
% size(y) % 5000 x 1

a3 = h_theta; % assigning into a3 for naming's sake.

d3 = a3 .- y_matrix; % 5000 x 10
d2 = d3 * Theta2 .* sigmoidGradient([ones(m, 1) z2]);

% size(d2(:, 2:end)) 5000 x 25

% my a1 is 5000 x 401; X is 5000 x 400

Theta1_grad = (1 / m) .* ( d2(:, 2:end)' * a1 );
Theta2_grad = (1 / m) .* ( d3' * a2 );

for t = 1:m

end

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
