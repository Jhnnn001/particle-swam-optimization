classdef Problem
    %PROBLEM Scalar objective and one [lower, upper] row per dimension.
    % problem = Problem(objective, bounds, 'maximize', false)
    % The objective receives a 1-by-D double row and returns a real scalar.

    properties (SetAccess = private)
        objective
        bounds
        maximize
        lb
        ub
        dim
    end

    methods
        function obj = Problem(objective, bounds, varargin)
            if ~isa(objective, 'function_handle')
                error('pso4:InvalidProblem', 'objective must be a function handle.');
            end
            validateattributes(bounds, {'numeric'}, {'real', '2d', 'nonempty', 'finite', 'ncols', 2});
            bounds = double(bounds);
            width = bounds(:, 2) - bounds(:, 1);
            if any(~isfinite(width) | width <= 0)
                error('pso4:InvalidProblem', 'bounds require finite lower < upper and finite widths.');
            end
            p = inputParser;
            p.PartialMatching = false;
            p.CaseSensitive = true;
            addParameter(p, 'maximize', false, @(x) islogical(x) && isscalar(x));
            parse(p, varargin{:});
            obj.objective = objective;
            obj.bounds = bounds;
            obj.maximize = p.Results.maximize;
            obj.lb = bounds(:, 1).';
            obj.ub = bounds(:, 2).';
            obj.dim = size(bounds, 1);
        end
    end
end
