% This code generates the datasets for the examples written in the folder
% "examples".
% To generate the dataset corresponding to the file helmholtz.m,
% run the command
% generate_gl_example('helmholtz');

function generate_data(D)
    % Add warning about Chebfun
    warning("This code requires the Chebfun package. See http://www.chebfun.org/download/ for installation details.")

    % Load the differential operator example
    % Parameter of the equation
    dom = [0,1];
    diff_op = chebop(@(x,u) -diff(u,2)+D*diff(u), dom);
    diff_op.bc = @(x,u) [u(dom(1)); u(dom(2))];
    
    % Get other parameters
    lambda = 0.03;
    
    % Number of sampled functions f
    Nsample = 100;

    % Training points for f
    Nf = 200;
    Y = linspace(dom(1), dom(2), Nf)';

    % Training points for u
    Nu = 200;
    X = linspace(dom(1), dom(2), Nu)';

    % Evaluation points for G
    NGx = 1000;
    NGy = 1000;
    XG = linspace(dom(1), dom(2), NGx)';
    YG = linspace(dom(1), dom(2), NGy)';
    
    % Define the Gaussian process kernel
    K = chebfun2(@(x,y)exp(-(x-y).^2/(2*lambda^2)), [0,1,0,1]);

    % Compute the Cholesky factorization of K
    L = chol(K, 'lower');
    
    % Setup preferences for solving the problem.
    options = solver_options();

    % Initialize data arrays
    U = zeros(Nu, Nsample, 1);
    F = zeros(Nf, Nsample, 1);

    % Loop over the number of sampled functions f
    for i = 1:Nsample
        sprintf("i = %d/%d",i, Nsample)
        
        % Sample from a Gaussian process
        f = generate_random_fun(L);
        rhs = f;

        % Solve the equation
        u = solvebvp(diff_op, rhs, options);
        
        % Evaluate at the training points
        U(:,i,:) = u(X);
        F(:,i,:) = rhs(Y);
    end
    
    % Compute homogeneous solution
    u_hom = solvebvp(diff_op, 0, options);    
    U_hom = u_hom(X);
    
    % Normalize the solution for homogeneous problems
    if all(iszero(u_hom))
        scale = max(abs(U),[],'all');
        U = U/scale;
        F = F/scale;
    end
    
    if abs(D) > 0
        ExactGreen = sprintf("np.exp(-%.2f*y)*(1-np.exp(%.2f*(x-1)))*(-1+np.exp(%.2f*y))*(x>=y)/(%.2f*(1-np.exp(-%.2f)))+"...
        +"np.exp(-%.2f*y)*(-1+np.exp(%.2f*x))*(1-np.exp(%.2f*(y-1)))*(x<y)/(%.2f*(1-np.exp(-%.2f)))", repmat(D,1,10));
    else
        ExactGreen = 'x*(1-y)*(x<=y) + y*(1-x)*(x>y)';
    end
        
    ExactGreen = convertStringsToChars(ExactGreen);
    % Save the data
    save(sprintf('data/advection_%.2f.mat',D),"X","Y","U","F","U_hom","XG","YG","ExactGreen")
    
end

function f = generate_random_fun(L)
% Take a cholesky factor L of a covariance kernel and return a smooth
% random function.

% Generate a vector of random numbers
u = randn(rank(L),1);
f = L*u;
end

function options = solver_options()
% Setup preferences for solving the problem.
% Create a CHEBOPPREF object for passing preferences.
% (See 'help cheboppref' for more possible options.)
options = cheboppref();

% Print information to the command window while solving:
options.display = 'iter';

% Option for tolerance.
options.bvpTol = 5e-13;

% Option for damping.
options.damping = false;

% Specify the discretization to use. Possible options are:
%  'values' (default)
%  'coeffs'
%  A function handle (see 'help cheboppref' for details).
options.discretization = 'values';

end