function [bestSolution, bestFunc, A, B] = TSA(func, lb, ub, Np, T, k, p)
% This function implements Toroidal Search Algorithm
% Developed by Changin Oh and Myongin Oh (2025-03-29)
%
% Inputs
%   func: objective function to minimize (function)
%   lb: lower bound (vector)
%   ub: upper bound (vector)
%   Np: size of population (positive integer)
%   T: number of iterations (positive integer)
%   k: exploration/exploitation balance constant (default = 2 -> 50%)
%   p: probability of toroidal search (default = 0.8 -> 80%)
%
% Outputs
%   bestSolution: best solution (vector)
%   bestFunc: best function value (scalar)
%   A: track of average function values (vector)
%   B: track of best function values (vector)

% Initialize parameter settings (when k and p are not provided)
if nargin == 5
    k = 2;
    p = 0.8;
end

% Set dimensionality
D = length(lb);

% Generate initial population
P = rand(Np, D).*(ub-lb) + lb;

% Evaluate function value of initial population
F = nan(Np, 1);
for i = 1:Np
    F(i) = func(P(i, :));
end

% Initialize winding vectors for toroidal movement tracking
W = zeros(Np, D);

% Initialize tracking arrays
A = nan(T + 1, 1);  % average function value
B = nan(T + 1, 1);  % best function value

% Determine best agent of initial population
[Fbest, ind] = min(F);
Pbest = P(ind, :);

% Store initial population statistics
A(1) = mean(F);
B(1) = min(F);

% Perform TSA
for t = 1:T
    % Define modified sigmoid function for search balance
    s = 1/(1+exp(100*(k*t/T-1)));
   
    for i = 1:Np
        % Select current agent
        Pi = P(i, :);

        % Generate angle vector
        theta = pi*rand(1, D);  

        % Perform toroidal search with probability p
        if rand < p
            % Choose random agent different from current agent
            candidates = [1:i-1 i+1:Np];
            j = candidates(randperm(Np-1, 1));
            Pj = P(j, :);
                       
            % Compute toroidal distance considering periodic boundary conditions
            r = rand;
            dist = max(abs((Pbest - r*Pi)), (ub-lb) - abs((Pbest - r*Pi)));

            % Define scaling factor based on winding vector magnitude
            scaling = 1 / (1 + norm(W(i, :)));

            % Find candidate using toroidal movement strategy
            Pnew = Pi + s*abs(cos(theta)).*sign(Pi).*dist + (1-s)*scaling*cos(theta).*(Pi-Pj);
        else
            % Find candidate using direct attraction
            r = rand+1;
            Pnew = Pi + abs(cos(theta)).*(Pbest - r*Pi);
        end
        
        % Identify out-of-bounds elements
        crossUb = Pnew > ub;  
        crossLb = Pnew < lb;
                        
        % Apply toroidal wrapping to keep solutions within bounds
        Pnew = lb + mod(Pnew - lb, ub - lb);
        
        % Evaluate new candidate solution
        Fnew = func(Pnew);
        
        % Perform greedy selection
        if Fnew < F(i)
            % Update winding vector to track toroidal movements
            W(i, :) = W(i, :) + crossUb - crossLb;  % Increment for upper bound crossings, decrement for lower bound crossings
            
            % Accept new solution
            P(i, :) = Pnew;
            F(i) = Fnew;
            
            % Update global best if new solution is better
            if Fnew < Fbest
                Pbest = Pnew;
                Fbest = Fnew;
            end
        end
    end
    
    % Store function values for analysis
    A(t+1) = mean(F);   % average function value
    B(t+1) = min(F);    % best function value
end

% Return final best solution and function value
[val, ind] = min(F);
bestSolution = P(ind, :);
bestFunc = val;
end