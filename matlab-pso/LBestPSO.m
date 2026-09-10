function result = LBestPSO(problem, varargin)
%LBESTPSO Ring PSO with decreasing inertia and velocity clamping.
% result = LBestPSO(problem, 'n_particles', 30, 'n_iter', 200, ...)
% Parameters: seed=[], boundary='reflect', callback=[], w_start=0.9,
% w_end=0.4, c1=1.6, c2=1.5, n_social=2, n_social_final=[], v_clamp=0.8.
% Reference: Engelbrecht (2007), Eqs. (16.8), (16.18), (16.26).
result = pso_optimize('LBestPSO', problem, varargin{:});
end
