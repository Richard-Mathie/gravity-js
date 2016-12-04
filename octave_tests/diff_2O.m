function [D, Xm1, Xp1] = diff_2O(X)

[N,M] = size(X);
 
Xm1 = [X(:,M), X(:, 1:M-1)];
Xp1 = [X(:, 2:M), X(:, 1)];

D = (Xp1 - Xm1)/2;
end