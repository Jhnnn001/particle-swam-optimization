function tests = test_pso
tests = functiontests(localfunctions);
end

function testRunContract(testCase)
bounds = [0 1; 100 101; -3 -2];
calls = 0;
seen = 0;
problem = Problem(@objective, bounds);
for name = {'GBestPSO', 'LBestPSO', 'EBPSO', 'QPSO'}
    calls = 0;
    seen = 0;
    optimizer = str2func(name{1});
    r = optimizer(problem, 'n_iter', 20, 'seed', 0, 'callback', @observe);
    verifyEqual(testCase, seen, 20);
    verifyEqual(testCase, r.n_evals, calls);
    verifyEqual(testCase, calls, 30 * 21 + 20 * strcmp(name{1}, 'EBPSO'));
    verifyEqual(testCase, r.n_iter, 20);
    verifyEqual(testCase, r.stop_reason, 'max_iter');
    verifySize(testCase, r.history, [21 1]);
    verifyTrue(testCase, all(diff(r.history) <= 0));
    verifyEqual(testCase, r.f, sum(r.x.^2));
    verifyEqual(testCase, r.f, r.history(end));
end
    function f = objective(x)
        assert(isa(x, 'double') && isequal(size(x), [1 3]));
        assert(all(x >= bounds(:, 1).' & x <= bounds(:, 2).'));
        calls = calls + 1;
        f = sum(x.^2);
    end
    function stop = observe(s)
        seen = seen + 1;
        verifyEqual(testCase, s.t, seen);
        verifyEqual(testCase, s.n_evals, calls);
        verifyEqual(testCase, s.values, sum(s.positions.^2, 2));
        verifyEqual(testCase, s.best_f, sum(s.best_x.^2));
        stop = false;
    end
end

function testRepeatabilityDirectionAndCallback(testCase)
problem = Problem(@(x) sum(x.^2), repmat([-5 5], 3, 1));
maximum = Problem(@(x) -sum(x.^2), problem.bounds, 'maximize', true);
global_state = rng;
for name = {'GBestPSO', 'LBestPSO', 'EBPSO', 'QPSO'}
    optimizer = str2func(['' name{1}]);
    r = optimizer(problem, 'n_iter', 20, 'seed', 3);
    again = optimizer(problem, 'n_iter', 20, 'seed', 3, 'callback', @mutate);
    high = optimizer(maximum, 'n_iter', 20, 'seed', 3);
    verifyEqual(testCase, r, again);
    verifyEqual(testCase, r.x, high.x);
    verifyEqual(testCase, r.positions, high.positions);
    verifyEqual(testCase, r.history, -high.history);
    different = optimizer(problem, 'n_iter', 20, 'seed', 4);
    verifyFalse(testCase, isequal(r.positions, different.positions));
    for stop_at = [3 5]
        stopped = optimizer(problem, 'n_iter', 5, 'seed', 0, 'callback', @(s) s.t == stop_at);
        verifyEqual(testCase, stopped.n_iter, stop_at);
        verifyEqual(testCase, stopped.stop_reason, 'callback');
        verifyEqual(testCase, numel(stopped.history), stop_at + 1);
    end
end
verifyEqual(testCase, rng, global_state);
    function stop = mutate(s)
        s.positions(:) = -999;
        s.values(:) = -999;
        s.best_x(:) = -999;
        stop = [];
    end
end

function testBoundariesAndMigration(testCase)
lb = [-2 100];
ub = [3 101];
width = ub - lb;
verifyEqual(testCase, pso_boundary(ub + 2.5 * width, lb, ub, 'reflect'), lb + .5 * width);
verifyEqual(testCase, pso_boundary(lb - .3 * width, lb, ub, 'reflect'), lb + .3 * width, 'AbsTol', 1e-13);
verifyEqual(testCase, pso_boundary(ub, lb, ub, 'wrap'), lb);
[x, f, target] = pso_migrate([0 0; .4 .4; 1 1], [7; 5; 1]);
verifyEqual(testCase, target, 1);
verifyEqual(testCase, x(1, :), [1 1]);
verifyEqual(testCase, f, [1; 5; 1]);
[x, f, target] = pso_migrate([0 0; .5 0; 1 0; 1 1], [7; 5; 3; 1]);
verifyEqual(testCase, target, 3);
verifyEqual(testCase, x(3, :), [1 1]);
verifyEqual(testCase, f, [7; 5; 1; 1]);
[~, ~, target] = pso_migrate(ones(4, 2), ones(4, 1));
verifyEmpty(testCase, target);
end

function testSeededMovementEquations(testCase)
problem = Problem(@(x) sum(x.^2), repmat([0 10], 2, 1));
stream = RandStream('mt19937ar', 'Seed', 1);
x = 10 * rand(stream, 5, 2);
[~, best] = min(sum(x.^2, 2));
r1 = rand(stream, size(x));
r2 = rand(stream, size(x));
expected = x + .7 * r2 .* (x(best, :) - x);
r = GBestPSO(problem, 'n_particles', 5, 'n_iter', 1, 'seed', 1, 'c2', .7);
verifyEqual(testCase, r.positions, expected, 'AbsTol', 1e-13);

% k=0 has only the particle itself; a growing ring reaches k=1 at t=2.
r = LBestPSO(problem, 'n_particles', 5, 'n_iter', 1, 'seed', 1, 'n_social', 0);
verifyEqual(testCase, r.positions, x);
stream = RandStream('mt19937ar', 'Seed', 1);
rand(stream, 5, 2);
rand(stream, 5, 2);
rand(stream, 5, 2);
rand(stream, 5, 2);
r2 = rand(stream, 5, 2);
social = [5; 1; 2; 3; 4];
for i = 1:5
    neighbors = mod((i - 1) + (-1:1), 5) + 1;
    [~, local] = min(sum(x(neighbors, :).^2, 2));
    social(i) = neighbors(local);
end
expected = x + .5 * r2 .* (x(social, :) - x);
r = LBestPSO(problem, 'n_particles', 5, 'n_iter', 2, 'seed', 1, ...
    'n_social', 0, 'n_social_final', 1, 'c2', .5);
verifyEqual(testCase, r.positions, expected, 'AbsTol', 1e-13);

stream = RandStream('mt19937ar', 'Seed', 1);
rand(stream, 5, 2);
r1 = 1 - rand(stream, 5, 2);
r2 = 1 - rand(stream, 5, 2);
u = 1 - rand(stream, 5, 2);
sgn = 2 * (rand(stream, 5, 2) > .5) - 1;
phi = 2 * r1 ./ (2 * r1 + r2);
expected = phi .* x + (1 - phi) .* x(best, :) + sgn .* .5 .* abs(mean(x, 1) - x) .* log(1 ./ u);
expected = pso_boundary(expected, [0 0], [10 10], 'reflect');
r = QPSO(problem, 'n_particles', 5, 'n_iter', 1, 'seed', 1, 'c1', 2, 'migration', false);
verifyEqual(testCase, r.positions, expected, 'AbsTol', 1e-13);
end

function testEBPSOSequentialMoveAndGreedyTrial(testCase)
problem = Problem(@(x) x(1), [0 10]);
stream = RandStream('mt19937ar', 'Seed', 2);
x = 10 * rand(stream, 2, 1);
personal = x;
g = min(x);
v = zeros(2, 1);
% Independent two-sweep calculation also exercises decreasing inertia.
for t = 1:2
    for i = 1:2
        branch = rand(stream);
        r1 = rand(stream);
        r2 = rand(stream);
        if branch < exp(-t / 10)
            wp = (1 + g) / (2 + personal(i) + g);
            wg = wp;
        else
            wp = exp(-abs(x(1) - x(2)) / 20);
            wg = 1 - wp;
        end
        v(i) = (.9 - .5 * t / 10) * v(i) + .2 * r1 * (wp * personal(i) - x(i)) + .3 * r2 * (wg * g - x(i));
        x(i) = pso_boundary(x(i) + v(i), 0, 10, 'reflect');
        personal(i) = min(personal(i), x(i));
        g = min(g, personal(i));
    end
    xi = (11 - t) / 10;
    trial = (1 - xi) * g + xi * 10 * rand(stream);
    g = min(g, trial);
end
r = EBPSO(problem, 'n_particles', 2, 'n_iter', 10, 'seed', 2, ...
    'c1', .2, 'c2', .3, 'callback', @(s) s.t == 2);
verifyEqual(testCase, r.positions, x, 'AbsTol', 1e-13);
verifyEqual(testCase, r.f, g, 'AbsTol', 1e-13);
verifyEqual(testCase, r.n_evals, 8);
end

function testValidationAndNonfiniteObjectives(testCase)
verifyError(testCase, @() Problem(@sum, [1 1]), 'pso4:InvalidProblem');
problem = Problem(@sum, [0 1]);
verifyError(testCase, @() QPSO(problem, 'c1', 0), 'pso4:InvalidOption');
verifyError(testCase, @() LBestPSO(problem, 'n_particles', 4), 'pso4:InvalidOption');
verifyError(testCase, @() EBPSO(problem, 'w_min', 1, 'w_max', .4), 'pso4:InvalidOption');
verifyError(testCase, @() GBestPSO(problem, 'boundary', 'clip'), 'pso4:InvalidOption');
for bad = {NaN, Inf, -Inf, [1 2], 1i, '1'}
    p = Problem(@(x) bad{1}, [0 1]);
    verifyError(testCase, @() GBestPSO(p), 'pso4:InvalidObjective');
end
stream = RandStream('mt19937ar', 'Seed', 0);
initial = rand(stream, 4, 1);
target = initial(3);
for maximize = [false true]
    p = Problem(@mostly_invalid, [0 1], 'maximize', maximize);
    r = GBestPSO(p, 'n_particles', 4, 'n_iter', 5, 'seed', 0);
    verifyEqual(testCase, r.x, target);
    verifyEqual(testCase, r.f, 7);
end
verifyError(testCase, @() GBestPSO(Problem(@broken, [0 1])), 'test:ObjectiveError');
    function f = mostly_invalid(x)
        f = NaN;
        if x == target
            f = 7;
        end
    end
    function f = broken(~)
        error('test:ObjectiveError', 'Solver failed.');
        f = 0; %#ok<UNRCH>
    end
end

function testConvergence(testCase)
for offset = [0 2.5]
    problem = Problem(@(x) sum((x - offset).^2), repmat([-5 5], 5, 1));
    for name = {'GBestPSO', 'LBestPSO', 'QPSO'}
        optimizer = str2func(['' name{1}]);
        options = {};
        if strcmp(name{1}, 'QPSO')
            options = {'migration', false};
        end
        r = optimizer(problem, 'n_iter', 300, 'seed', 0, options{:});
        verifyLessThan(testCase, r.f, 1e-2);
    end
end
r = EBPSO(Problem(@(x) sum(x.^2), repmat([-5 5], 5, 1)), 'n_iter', 300, 'seed', 0);
verifyLessThan(testCase, r.f, 1e-2);
end
