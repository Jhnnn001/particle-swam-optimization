function result = EBPSO(problem, varargin)
%EBPSO Sequential empirical-balance PSO with a greedy global-best trial.
% result = EBPSO(problem, 'n_particles', 30, 'n_iter', 200, ...)
% Parameters: seed=[], boundary='reflect', callback=[], w_min=0.4,
% w_max=0.9, c1=1.4, c2=1.3, v_clamp=1.0.
% Decreasing inertia follows the prose; both printed weights use fit(pbest).
% Reference: Zhang & Kong (2023), Eqs. (4)-(9), Algorithm 1.
result = pso_optimize('EBPSO', problem, varargin{:});
end
