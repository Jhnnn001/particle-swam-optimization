function result = pso_optimize(variant, problem, varargin)
% Shared run bookkeeping; each switch branch preserves its update order.
if ~isa(problem, 'Problem') || ~isscalar(problem)
    error('pso4:InvalidProblem', 'problem must be a scalar Problem.');
end
defaults = struct('n_particles', 30, 'n_iter', 200, 'seed', [], ...
    'boundary', 'reflect', 'callback', []);
switch variant
    case 'GBestPSO'
        specific = struct('w', .7298, 'c1', 1.49618, 'c2', 1.49618);
        defaults.boundary = 'wrap';
    case 'LBestPSO'
        specific = struct('w_start', .9, 'w_end', .4, 'c1', 1.6, 'c2', 1.5, ...
            'n_social', 2, 'n_social_final', [], 'v_clamp', .8);
    case 'EBPSO'
        specific = struct('w_min', .4, 'w_max', .9, 'c1', 1.4, 'c2', 1.3, 'v_clamp', 1);
    case 'QPSO'
        specific = struct('alpha_start', 1, 'alpha_end', .5, 'c1', 1, 'c2', 1, 'migration', true);
end
p = inputParser;
p.PartialMatching = false;
p.CaseSensitive = true;
names = fieldnames(defaults);
for j = 1:numel(names)
    addParameter(p, names{j}, defaults.(names{j}));
end
names = fieldnames(specific);
for j = 1:numel(names)
    addParameter(p, names{j}, specific.(names{j}));
end
parse(p, varargin{:});
o = p.Results;
names = fieldnames(o);
for j = 1:numel(names)
    name = names{j};
    if ~ismember(name, {'seed', 'boundary', 'callback', 'migration', 'n_social_final'})
        validateattributes(o.(name), {'numeric'}, {'real', 'scalar', 'finite', 'nonnegative'}, variant, name);
        o.(name) = double(o.(name));
    end
end
validateattributes(o.n_particles, {'double'}, {'integer', '>=', 2}, variant, 'n_particles');
validateattributes(o.n_iter, {'double'}, {'integer', '>=', 1}, variant, 'n_iter');
if ~isempty(o.seed)
    validateattributes(o.seed, {'numeric'}, {'real', 'scalar', 'integer', '>=', 0, '<=', 2^32 - 1}, variant, 'seed');
end
if ~((ischar(o.boundary) && isrow(o.boundary)) || (isstring(o.boundary) && isscalar(o.boundary))) ...
        || ~any(strcmp(o.boundary, {'reflect', 'wrap'}))
    error('pso4:InvalidOption', 'boundary must be ''reflect'' or ''wrap''.');
end
if ~isempty(o.callback) && ~isa(o.callback, 'function_handle')
    error('pso4:InvalidOption', 'callback must be a function handle or [].');
end
if isfield(o, 'v_clamp') && o.v_clamp <= 0
    error('pso4:InvalidOption', 'v_clamp must be positive.');
end
switch variant
    case 'LBestPSO'
        validateattributes(o.n_social, {'double'}, {'integer'}, variant, 'n_social');
        if o.w_start < o.w_end || o.v_clamp > 1 || 2 * o.n_social + 1 > o.n_particles
            error('pso4:InvalidOption', 'Require w_start >= w_end, v_clamp <= 1 and 2*n_social+1 <= n_particles.');
        end
        if ~isempty(o.n_social_final)
            validateattributes(o.n_social_final, {'numeric'}, {'real', 'scalar', 'finite', 'integer', '>=', o.n_social});
            o.n_social_final = double(o.n_social_final);
            if 2 * o.n_social_final + 1 > o.n_particles || o.n_social_final - o.n_social + 1 > o.n_iter
                error('pso4:InvalidOption', 'Ring growth requires valid half-widths and at most n_iter stages.');
            end
        end
    case 'EBPSO'
        if o.w_min > o.w_max
            error('pso4:InvalidOption', 'w_min must be <= w_max.');
        end
    case 'QPSO'
        if o.alpha_start < o.alpha_end || o.alpha_end <= 0 || o.c1 <= 0 || o.c2 <= 0
            error('pso4:InvalidOption', 'Require alpha_start >= alpha_end > 0 and c1, c2 > 0.');
        end
        if ~islogical(o.migration) || ~isscalar(o.migration)
            error('pso4:InvalidOption', 'migration must be a logical scalar.');
        end
end

seed = o.seed;
if isempty(seed)
    seed = 'shuffle';
else
    seed = double(seed);
end
stream = RandStream('mt19937ar', 'Seed', seed);
n = o.n_particles;
T = o.n_iter;
lb = problem.lb;
ub = problem.ub;
width = ub - lb;
direction = 1 - 2 * double(problem.maximize);
n_evals = 0;
x = lb + rand(stream, n, problem.dim) .* width;
f = evaluate(x);
if ~any(isfinite(f))
    error('pso4:InvalidObjective', 'objective returned no finite value during initialization.');
end
pbest = x;
pbest_f = f;
[gbest_f, best] = min(f);
gbest = x(best, :);
v = [];
if ~strcmp(variant, 'QPSO')
    v = zeros(size(x));
end
history = zeros(T + 1, 1);
history(1) = gbest_f;
reason = 'max_iter';
for t = 1:T
    switch variant
        case {'GBestPSO', 'LBestPSO'}
            if strcmp(variant, 'GBestPSO')
                w = o.w;
                social = gbest;
            else
                w = o.w_start - (t / T) * (o.w_start - o.w_end);
                k = o.n_social;
                if ~isempty(o.n_social_final)
                    k = k + floor((t - 1) * (o.n_social_final - k + 1) / T);
                end
                neighbors = mod((0:n-1).' + (-k:k), n) + 1;
                [~, local] = min(reshape(pbest_f(neighbors), size(neighbors)), [], 2);
                indices = neighbors(sub2ind(size(neighbors), (1:n).', local));
                social = pbest(indices, :);
            end
            r1 = rand(stream, size(x));
            r2 = rand(stream, size(x));
            v = w * v + o.c1 * r1 .* (pbest - x) + o.c2 * r2 .* (social - x);
            if strcmp(variant, 'LBestPSO')
                vmax = o.v_clamp * width;
                v = min(max(v, -vmax), vmax);
            end
            x = pso_boundary(x + v, lb, ub, o.boundary);
            f = evaluate(x);
            update_bests(1:n);
        case 'EBPSO'
            w = o.w_max - (o.w_max - o.w_min) * t / T;
            probability = exp(-t / T);
            vmax = o.v_clamp * width;
            for i = 1:n
                branch = rand(stream);
                r1 = rand(stream);
                r2 = rand(stream);
                if branch < probability
                    a = fit(pbest_f(i));
                    b = fit(gbest_f);
                    scale = max(a, b);
                    wp = (a / scale) / (a / scale + b / scale);
                    wg = wp; % Both numerators use fit(pbest), as printed.
                else
                    sigma = mean(sqrt(sum((x - mean(x, 1)).^2, 2))) / norm(width);
                    wp = exp(-sigma);
                    wg = 1 - wp;
                end
                v(i, :) = w * v(i, :) + o.c1 * r1 * (wp * pbest(i, :) - x(i, :)) ...
                    + o.c2 * r2 * (wg * gbest - x(i, :));
                v(i, :) = min(max(v(i, :), -vmax), vmax);
                x(i, :) = pso_boundary(x(i, :) + v(i, :), lb, ub, o.boundary);
                f(i) = evaluate(x(i, :));
                update_bests(i);
            end
            xi = (T - t + 1) / T;
            target = lb + rand(stream) * width;
            trial = (1 - xi) * gbest + xi * target;
            value = evaluate(trial);
            if value < gbest_f
                gbest = trial;
                gbest_f = value;
            end
        case 'QPSO'
            alpha = o.alpha_start - (t / T) * (o.alpha_start - o.alpha_end);
            mean_best = mean(pbest, 1);
            r1 = 1 - rand(stream, size(x));
            r2 = 1 - rand(stream, size(x));
            scale = max(o.c1, o.c2);
            a = (o.c1 / scale) * r1;
            b = (o.c2 / scale) * r2;
            phi = a ./ (a + b); % Eq. (7), not directly uniform phi.
            u = 1 - rand(stream, size(x));
            jump_sign = 2 * (rand(stream, size(x)) > .5) - 1;
            attractor = phi .* pbest + (1 - phi) .* gbest;
            x = pso_boundary(attractor + jump_sign .* alpha .* abs(mean_best - x) .* log(1 ./ u), lb, ub, o.boundary);
            f = evaluate(x);
            update_bests(1:n);
            if o.migration
                [x, f] = pso_migrate(x, f);
            end
    end
    history(t + 1) = gbest_f;
    if ~isempty(o.callback)
        snapshot = struct('t', t, 'positions', x, 'values', direction * f, ...
            'best_x', gbest, 'best_f', direction * gbest_f, 'n_evals', n_evals);
        stop = o.callback(snapshot);
        if islogical(stop) && isscalar(stop) && stop
            reason = 'callback';
            break
        end
    end
end
result = struct('x', gbest, 'f', direction * gbest_f, 'history', direction * history(1:t+1), ...
    'positions', x, 'n_iter', t, 'n_evals', n_evals, 'stop_reason', reason);

    function values = evaluate(rows)
        values = zeros(size(rows, 1), 1);
        for row = 1:size(rows, 1)
            n_evals = n_evals + 1;
            value = problem.objective(rows(row, :));
            if ~isnumeric(value) || ~isreal(value) || ~isscalar(value)
                error('pso4:InvalidObjective', 'objective must return a real numeric scalar.');
            end
            if isfinite(value)
                values(row) = direction * double(value);
            else
                values(row) = Inf;
            end
        end
    end

    function update_bests(indices)
        improved = indices(f(indices) < pbest_f(indices));
        pbest(improved, :) = x(improved, :);
        pbest_f(improved) = f(improved);
        [value, index] = min(pbest_f);
        if value < gbest_f
            gbest = pbest(index, :);
            gbest_f = value;
        end
    end
end

function value = fit(f)
if f > 0
    value = 1 / (1 + f);
else
    value = 1 + abs(f);
end
end
