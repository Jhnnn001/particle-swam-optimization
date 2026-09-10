function x = pso_boundary(x, lb, ub, mode)
% Position repair leaves velocities unchanged.
if any(~isfinite(x(:)))
    error('pso4:NumericalOverflow', 'Non-finite position; rescale bounds or reduce coefficients.');
end
width = ub - lb;
if strcmp(mode, 'wrap')
    x = lb + mod(x - lb, width);
else
    y = mod(x - lb, 2 * width);
    x = lb + min(y, 2 * width - y);
end
if any(~isfinite(x(:)))
    error('pso4:NumericalOverflow', 'Boundary arithmetic overflow; rescale bounds.');
end
end
