# Author: Chandra S. Nepali
# Date: 12-03-2016
# Gradient Descent, Linear regression

clear ; close all; clc

#-----------------------------------------------------------
function J = costValue(X, Y, theta);
    m = length(Y);
    J = (1/(2*m)) * (X*theta - Y)'*(X*theta - Y);
endfunction

function theta = gradientDescent(X, Y, theta, alpha);
    m = length(Y);
    theta = theta - (alpha/m) * X'*(X*theta - Y);
endfunction

function X_norm = normalize(X);
    X_norm = X;
    for i = 2:size(X, 2);
      X_norm(:,i) = X_norm(:,i) - mean(X(:,i));
      X_norm(:,i) = X_norm(:,i) / std(X(:,i));
    endfor    
endfunction  

# analytical method
function theta = normalEqu(X, Y);
    theta = zeros(size(X, 2), 1);
    theta = pinv(X'*X)*X'*Y;
endfunction  
#-----------------------------------------------------------

data = load('ex1data2.txt');
nrows = size(data, 1);
ncols = size(data, 2);

X = data(:, 1:ncols-1);          # features matrix
Y = data(:, ncols);              # price matrix

X = [ones(nrows,1), X];         # add a column of 1s
theta = zeros(ncols, 1);        # initialize theta
X = normalize(X);

niter = 1500;
alpha = 0.01;

J_hist = zeros(niter, 1);
theta_hist = zeros(niter, length(theta));
i_hist = zeros(niter, 1);
J = 0;
for i = 1:niter;
    J = costValue(X, Y, theta);
    theta = gradientDescent(X, Y, theta, alpha);
    J_hist(i) = J;
    for j = 1:length(theta);
      theta_hist(i,j) = theta(j,:);
    endfor    
    i_hist(i) = i;
endfor

# plot cost vs number of interation
figure;
plot(i_hist, J_hist, 'Linewidth', 2);
xlabel('number of interation');
ylabel('cost');

# plot all theta vs number of interation
for i  = 1: length(theta);
    figure;
    plot(i_hist, theta_hist(:,i), 'Linewidth', 2);
    xlabel('number of interation');
    st = int2str(i);
    lb = strcat('\theta_', st);
    ylabel(lb);
endfor

printf('---- Gradient desent values of thetas ----\n');
theta

printf('---- analytical values of thetas ----\n');
theta_norm = normalEqu(X, Y)

printf('---- differences ----\n');
for i = 1:length(theta);
    diff = abs(theta(i) - theta_norm(i))*100.0/abs(theta_norm(i));
    st = int2str(i);
    lb = strcat('\theta_', st);
    printf('diff = %d %s\n', diff, '%');
endfor
