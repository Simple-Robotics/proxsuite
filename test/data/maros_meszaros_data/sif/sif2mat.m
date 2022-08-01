function[] = sif2mat(probname, decodecmd) %
             Decode problem fprintf('Decoding %s...', probname);
[ status, cmdout ] = system(sprintf('%s sif_data/%s', decodecmd, probname));
if status
  fprintf(cmdout);
error('Decoding failed!');
end fprintf('[Done]\n');

% Load problem problem = cutest_setup();

% Extract data n = problem.n;
% Number of variables mc = problem.m;
% Number of
  constraints(no bounds) m = mc + n;
% Number of
  constraints(including bounds) x0 = problem.x + 1;
% Initial guess of primal
  solution(used to evaluate gradients) v0 = problem.v + 1;
% Initial guess of dual solution

  % Objective P = sparse(cutest_sphess(x0, v0));
% Hessian(TODO : Use cutest_sphess) Px = cutest_hprod(x0);
% P* x q = cutest_grad(x0) - Px;
% q

  % Test objective x_test = ones(n, 1);
obj_cutest = cutest_obj(x_test);
r = cutest_obj(zeros(n, 1));
obj_rewritten = .5 * x_test ' * P * x_test + q' * x_test + r;
if norm (obj_cutest - obj_rewritten)
  > 1e-6 error('Objective function decoding failed') end

      % Constraints cl = problem.cl;
cu = problem.cu;

[ cx0, A_c ] = cutest_scons(x0);  % Compute Jacbian
% NOTE: Need to adjust constraints to take into account the shift
% For equality constraints, for example, the following holds
% c(x) = 0 => c(x0) + J(x0)(x - x0) = c(x0) - A * x0 + A * x = 0
% c(x) = 0 => A * x = A * x0 - c(x0)
cons_shift = A_c * x0 - cx0;
cl = cl + cons_shift;
cu = cu + cons_shift;
l = [cl; problem.bl];
u = [cu; problem.bu];

% [ ~, A ] = cutest_lagjac(x0);
% Compute Jacobian A = [sparse(A_c); speye(n)];

% Store data to mat file
    save(sprintf('../%s', probname), 'm', 'n', 'P', 'q', 'r', 'A', 'l', 'u');

% Solve with gurobi(debug) % x = sdpvar(n, 1);
% objective = 0.5 * x ' * P * x + q' * x + r;
% constraints = [l <= A * x; A * x <= u];
% options = sdpsettings('solver', 'gurobi', 'verbose', 1);
% sol_gurobi = optimize(constraints, objective, options);

%
  Unload problem
  cutest_terminate()

  % Cleanup directory cleanup_dir();

end
