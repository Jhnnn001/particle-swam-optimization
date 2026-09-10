function result = QPSO(problem, varargin)
%QPSO Equation (7) QPSO with optional Algorithm 2 diversity migration.
% result = QPSO(problem, 'n_particles', 30, 'n_iter', 200, ...)
% Parameters: seed=[], boundary='reflect', callback=[], alpha_start=1.0,
% alpha_end=0.5, c1=1.0, c2=1.0, migration=true.
% Reference: Gong et al. (2024), Eqs. (6), (7), (9), (10), Algorithm 2.
result = pso_optimize('QPSO', problem, varargin{:});
end
