function [x, f, target] = pso_migrate(x, f)
% Algorithm 2: exact coordinate mismatch counts on current positions.
n = size(x, 1);
h = zeros(n, 1);
% O(N^2 D) mismatch counts over the whole swarm.
for i = 1:n
    h(i) = nnz(x ~= x(i, :)) / (n - 1);
end
[~, best] = min(f);
if numel(unique(h)) <= 2
    [~, target] = max(f);
else
    [~, target] = min(h);
end
if target == best
    target = [];
else
    x(target, :) = x(best, :);
    f(target) = f(best);
end
end
