function result = GBestPSO(problem, varargin)
%GBESTPSO Synchronous global-best PSO with constant inertia.
% result = GBestPSO(problem, 'n_particles', 30, 'n_iter', 200, ...)
% Parameters: seed=[], boundary='wrap', callback=[],
% w=0.7298, c1=1.49618, c2=1.49618. See README.md for the result contract.
% Reference: Engelbrecht (2007), Eqs. (16.1), (16.22).
result = pso_optimize('GBestPSO', problem, varargin{:});
end
