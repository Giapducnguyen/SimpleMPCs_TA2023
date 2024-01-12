% % Example 1: Tracking Problem -Linear MPC with Observers
close all; clear all; clc

%% Continuous-time double integrator dynamics
Ac = [0,1;0,0];
Bc = [0;1];
Cc = [1,0];
Dc = 0;

[nx,nu] = size(Bc);     % number of states
ny = size(Cc,1);        % number of outputs
nr = 1;                 % number of references
nd = 1;                 % number of disturbances

%% Discretization
Ts = 0.1;               % sampling period
sysd = c2d(ss(Ac,Bc,Cc,Dc),Ts); % discretization
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Extended-state dynamics
% % First, extend the system state to include disturbance estimate w_hat(k)
% % z(k) = [x(k); w_hat(k)]
% % z(k+1) = Az * z(k) + Bz * u(k)
% % y(k) = Cd*x(k) = Cz * z(k)

Az = [Ad,                       Bd;
      zeros(nd,size(Ad,2)),     eye(nd)];
Bz = [Bd; zeros(nd,size(Bd,2))];
Cz = [Cd, zeros(size(Cd,1),nd)];

nz = size(Az,2); % number of extended state z

% % Second, extend system state to include previous control u(k-1) and
% % reference r(k);
% % Note that, now control input is du(k) = u(k)-u(k-1)

% % Extended state definition:
% % z_ext(k) = [z(k); u(k-1); r(k)] = [x(k); w_hat(k); u(k-1); r(k)]
% % Extended state dynamics:
% % z_ext(k+1) = Az_ext * z_ext(k) + Bz_ext * du(k)
% % Error as the output:
% % e(k) = Cd*x(k) - r(k) = Cz * z(k) - r(k) = E * z_ext(k)

Az_ext = [Az,                       Bz,             zeros(size(Az,1),1);
          zeros(nu,size(Az,2)),     eye(nu),        zeros(nu,1);
          zeros(nr,size(Az,2)),     zeros(nr,nu),   eye(nr)             ];
Bz_ext = [Bz; eye(nu); zeros(1,nu)];
E = [Cz, zeros(size(Cz,1),1), -eye(nr)];

%% State constraints
x1_min = -1;    x1_max = 1.1;
x2_min = -0.2;  x2_max = 0.2;

%% Control constraints
u_min = -0.1;       u_max = 0.1;        % limitations on magnitude
du_min = -0.08;     du_max = 0.08;      % limitations on rate of change

%% State, Control constraint Transformation to the Extended State
% % We transform the constraints of the original states and control inputs
% % to those of the extended states
% % We will use H-representation of contraints

% % (a) State constraints

% % Polyhedron matrices of original state constraints:       Fx * x <= fx
Fx = [1, 0;
      0, 1;
     -1  0;
      0 -1]; 
fx = [x1_max;x2_max;-x1_min;-x2_min];

% % Extend to Polyhedron matrices of z:                     Fz * z <= fz
Fz = Fx*[eye(nx), zeros(nx,nd)];
fz = fx;

% % Extend to Polyhedron matrices of z_ext:       Fz_ext * z_ext <= fz_ext
Fz_ext = Fz*[eye(nz), zeros(nz,nu), zeros(nz,nr)];
fz_ext = fx;

% % (b) Control constraints

% % Polyhedron matrices of original control input constraints: Gu * u <= gu
Gu = [1;-1]; 
gu = [u_max;-u_min];

% % % Extend to Polyhedron matrices of z_ext:       Gu_ext * z_ext <= gu_ext
% % z_ext(k) = [z(k); u(k-1); r(k)]
% % u(k-1) = [zeros(nu,z), eye(nu), zeros(nu,nr)] * z_ext
% % Gu * u(k-1) <= gu is equivalent to
% % Gu * ([zeros(nu,z), eye(nu), zeros(nu,nr)] * z_ext) <= gu
Gu_ext = Gu*[zeros(nu,nz), eye(nu), zeros(nu,nr)];
gu_ext = gu;

% % (c) Control rate constraints

% % These constraints are imposed directly to the decision variable du.
% % So matrix constraints are specified here without transformation:
Kdu = [1; -1];
kdu = [du_max; -du_min];

%% MPC data
Np = 15;            % Prediction horizon
R = 1; Qe = 1;      % Weighting matrices on control rate on tracking error
Q = E'*Qe*E;        % Transform weighting matrix on tracking error to extended state

%% Admissible (feasible) Decision Variable
% % Basically transform all state, control constraints to constraints of
% % the decision variables dU:
% % Uad.A * dU <= Uad.b - Uad.B * z_ext(0)
% % Uad.Ae * dU = Uad.be (not used, specified for calling MATLAB solver only)
Uad = AdmissibleInputs(Az_ext,Bz_ext,Np,Fz_ext,fz_ext,Gu_ext,gu_ext, Kdu, kdu);

%% Quadratic Programming Formulation
% % Weighting matrices on Np-horizon
Q_bar = blkdiag(kron(eye(Np-1),Q),Q);
R_bar = kron(eye(Np),R);

% % QP matrices
% % Transform objective function into QP format:
% % J = 0.5 * dU' * H * dU + q' * dU
% % Check lecture slides for the transformation!

[A_bar,B_bar] = genConMat(Az_ext,Bz_ext,Np);
H = B_bar'*Q_bar*B_bar + R_bar; % Quadratic term

% % mpc Activeset solver option settings
options = mpcActiveSetOptions;
iA0 = false(size(Uad.b));

%% Observer Gains Calculation
% % We present here 2 observers:

%

% % -----  Observer 1   -----

% % One: is state-disturbance estimation from OUTPUT feedback
% % x_hat(k+1) = Ad*x_hat(k) + Bd*u(k) + Bd*w_hat(k) + L1*(y(k)-Cd*x_hat(k))
% % w_hat(k+1) = w_hat(k) + L2*(y(k)-Cd*x_hat(k))

% % Or compactly,

% % xhat_ext(k) = [x_hat(k); w_hat(k)]
% % xhat_ext(k+1) = [Ad, Bd; zeros(nd,nx), eye(nd)]*xhat_ext(k) 
% %                 + [Bd; zeros(nd,nu)]*u(k) + L*Cd*(x(k)-x_hat(k))
% % where L = [L1; L2]

Qkf = diag([1e-8 1e-5 1e-4]);   % Process noise variance
Rkf = 5e-2;                     % Measurement noise variance
% Qkf = diag([1e-7 5e-5 5e-5]); Rkf = 5e-3;

A_ext = [Ad, Bd; zeros(nd,nx), eye(nd)];
C_ext = [Cd, zeros(ny,nd)];
Nkf = zeros(nx+nd, ny);
Gd = diag([1 1 1]);         % apply process noises to all states
Hd = zeros(ny, nx+nd);      % no direct transmission of noises
Gmodel = ss(A_ext, [[Bd;0] Gd], C_ext, [Dd Hd], Ts);

% % Options to calculate observer gains for Observer 1:
% % Option 1: Use MATLAB's Kalman filter function
[~, Kkf, Pkf] = kalman(Gmodel, Qkf, Rkf, Nkf, 'current');
L1 = Kkf(1:nx); L2 = Kkf(end);

% % Option 2: Use discrete-time Algebraic Riccati Equation
% A_ext = [Ad, Bd; zeros(nd,nx), eye(nd)];
% C_ext = [Cd, zeros(ny,nd)];
% [Pkf,Kkf] = idare(A_ext',C_ext',Qkf,Rkf,[],[]);
% L1 = Kkf(1:nx)'; L2 = Kkf(end)';

%}

%

% % -----    Observer 2   -----
% % This observer requires STATE-feedback, see more at the Function Helper
% % section (end of this file)

% % Observer gains selection:

% L = 0.5*[0, 5];                  % Tuning Option 1: by Trial-and-error

% L = (place(eye(nd),Bd', 0.4))'; % tuning Option 2: from stability analysis

M = eye(nx);
L = (eye(1) - 0.4)*pinv(M*Bd)*M;

%}

%% Main Simulation Loop
Nsim = 400;     % Number of simulation steps

% % Initialization
infeaOCP = 0;      % Infeasible OCP counter
x0 = [0;0];        % Initial state
u = 0;             % Initial control             
r0 = 1;            % Unit reference

x_hat = x0; w_h = 0;     % initialization for Disturbance Observer 1
p = 0;      w_h = 0;         % initialization for Disturbance Observer 2

% % State logging
X_log = zeros(nx,Nsim);             % original state logging
X_log(:,1) = x0;

Xext_log = zeros(nz+nu+nr,Nsim);    % extended state logging
x = x0;

% % Control logging
U_log = zeros(nu,Nsim);
U_log(:,1) = u;

% % Control rate logging
dU_log = zeros(nu,Nsim);

% % Reference logging
R_log = zeros(nr,Nsim);

% % Disturbance & disturbance estimation loggings
W_log = zeros(1,Nsim);
What_log = zeros(1,Nsim);

% % Execution time logging
ExecutionTime = nan(1,Nsim);


% % Main loop
for i = 1:Nsim
    % Disturbance & reference assignment
    if (i-1)*Ts < 1
        r = 0;
        w = 0;
    elseif (i-1)*Ts < 10
        r = r0;
        w = 0;
    elseif (i-1)*Ts < 25
        r = 0;
    else
        r = 0;
        w = 0.08;
    end

    % % Observation
    % % SELECT an Observer: 1 or 2
    obs_selection = 2;

    if obs_selection == 1
        % % Observer 1:
        y = Cd*x + Dd*u + 0.1*randn; % measurement noise added
        [x_hat, w_h] = KalmanDisObs(x_hat,w_h,u,y,Ad,Bd,Cd,L1,L2);
    else
        % % Observer 2:
        [p,w_h] = ssObserver(L,p,w_h,x,u,Ad,Bd);
    end

    % % Extended state logging
    Xext_log(:,i) = [x;w_h;u;r];

    % % QP matrix (linear term) computation
    q = B_bar'*Q_bar*A_bar*Xext_log(:,i);

    % % Calling solver and solve for optimal control increment du*
    tic;
    [dU, exitflag, iA0, ~] = mpcActiveSetSolver(H,q,Uad.A,Uad.b-Uad.B*Xext_log(:,i),Uad.Ae,Uad.be,iA0,options);
    % [dU,~,exitflag,~] = quadprog(H,q,Uad.A,Uad.b-Uad.B*Xext_log(:,i),Uad.Ae,Uad.be,[],[],double(iA0),options);
    ExecutionTime(i) = toc;

    % % If solver return infeasible OCP, apply zero move: du* = 0 and
    % % increase counter by 1
    if exitflag <= 0
        du = 0; infeaOCP = infeaOCP + 1;
    else    % Take the first optimal control du*
        du = dU(1,nu);
    end
    
    % % Compute actual control input = previous control + du*:
    u = u + du;
    
    % % Loggings
    U_log(:,i) = u;             % control
    dU_log(:,i) = du;           % control rate
    R_log(:,i) = r;             % reference
    W_log(:,i) = w;             % disturbance
    What_log(:,i) = w_h;        % disturbance estimation
    X_log(:,i) = x;             % state

    % % Propagate system forward
    x = Ad*x + Bd*(u + w + 0.01*randn); % process noise added
end

%% Plotting
lw = 1.5;
TimeArray = 0:Ts:(Ts*Nsim);

figure(1);
fig1 = tiledlayout(5,1);

nexttile
plot(TimeArray(1,1:end-1),X_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),R_log(1,:),'LineStyle','--','Color','b','LineWidth',lw);
plot(TimeArray(1,1:end-1),x1_max*ones(size(X_log(1,:))),'k--','LineWidth',lw)
ylabel('State $x_1$','interpreter','latex')
legend({'$x_1$','reference','constraints'},'interpreter','latex')
ylim([-0.1 1.2])
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray(1,1:end-1),X_log(2,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),x2_max*ones(size(X_log(2,:))),'k--','LineWidth',lw);
plot(TimeArray(1,1:end-1),x2_min*ones(size(X_log(2,:))),'k--','LineWidth',lw);
ylabel('State $x_2$','interpreter','latex')
legend({'$x_2$','constraints'},'interpreter','latex')
ylim([-0.22 0.22])
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
stairs(TimeArray(1,1:end-1),U_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
stairs(TimeArray(1,1:end-1),dU_log(1,:),'LineStyle','-','Color','b','LineWidth',lw);
plot(TimeArray(1,1:end-1),u_max*ones(size(U_log(1,:))),'k--','LineWidth',lw);
plot(TimeArray(1,1:end-1),du_max*ones(size(dU_log(1,:))),'g--','LineWidth',lw);
plot(TimeArray(1,1:end-1),u_min*ones(size(U_log(1,:))),'k--','LineWidth',lw);
plot(TimeArray(1,1:end-1),du_min*ones(size(dU_log(1,:))),'g--','LineWidth',lw);
ylabel('Control','interpreter','latex')
legend({'$u$','$\Delta u$','constraints on $u$','constraints on $\Delta u$'},'interpreter','latex')
ylim([-0.12 0.12])
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
hold on
plot(TimeArray(1,1:end-1),What_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
plot(TimeArray(1,1:end-1),W_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('Disturbance $w$','interpreter','latex')
% xlabel('Time (seconds)','interpreter','latex')
legend({'$\hat{w}$, estimate','$w$, true'},'interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray(1,1:end-1),1000.*ExecutionTime(1,:),'LineStyle','-','Color','m','LineWidth',lw);
xlabel('Time [s]','interpreter','latex')
legend({'Execution time [ms]'},'interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

fig1.TileSpacing = 'compact';
fig1.Padding = 'compact';
set(gcf,'Units','points','position',[400, 50,800, 700])

%% Function Helpers

function [A_bar,B_bar] = genConMat(A,B,Np)
% % This function returns matrices A_bar and B_bar satisfying:

% % Z_ext = A_bar * z_ext(0) + B_bar * dU

% % with:
% % Z_ext   =   [z_ext(1);  z_ext(2);   ...;    z_ext(Np)]
% % dU      =   [du(0);     du(1);      ...;    du(Np-1)]
% % Np      :   prediction horizon
% % z_ext(0):   current extended state
% % z_ext(0) =  [x(0); w_hat(0); u(-1); r(0)];

% % We present here 2 ways to create A_bar and B_bar as follows:

% % Method 1: (compact)
%
% A_bar = cell2mat(cellfun(@(x)A^x,num2cell((1:Np)'),'UniformOutput',false));
% B_bar = tril(cell2mat(cellfun(@(x)A^x,num2cell(toeplitz(0:Np-1)),'UniformOutput',false)))*kron(eye(Np),B);

% % Method 2: (using cell function)
%
A_bar = cell(Np, 1);
B_bar = cell(Np,Np);
b0 = zeros(size(B));
for i = 1:Np
    A_bar{i} = A^i;
    for j = 1:Np
        if i >= j
            B_bar{i,j} = A^(i-j)*B;
        else
            B_bar{i,j} = b0;
        end
    end
end
A_bar = cell2mat(A_bar);
B_bar = cell2mat(B_bar);
%
end

function Uad = AdmissibleInputs(A,B,Np,Fz_ext,fz_ext,Gu_ext,gu_ext, Kdu, kdu)
% % This function represent all constraints (state and control input) in
% % terms of decision variable dU.
% % This function returns an addmissible (feasible) decision variable dU
% % satisfying:
% % Uad.A * dU <= Uad.b - Uad.B * z_ext(0)
% % Uad.Ae * dU = Uad.be (not used, specified for calling MATLAB solver only)

% % (a) On state constraints:

% % One-step state constraint representation:
% %     Fz_ext * z_ext <= fz_ext

% % Np-horizon state constraint representation:
% %     blkdiag(Fz_ext, ..., Fz_ext) * Z_ext <= [fz_ext; ...; fz_ext]

% % Replace Z_ext = A_bar * z_ext(0) + B_bar * dU, we obtain:
% % --> blkdiag(Fz_ext, ..., Fz_ext) * (A_bar*z_ext(0) + B_bar*dU) 
% %     <= [fz_ext; ...; fz_ext]

% % --> blkdiag(Fz_ext, ..., Fz_ext) * B_bar*dU <= [fz_ext; ...; fz_ext]
% %     - blkdiag(Fz_ext, ..., Fz_ext) * A_bar*z_ext(0)

% % (b) On control constraints:

% % Similarly, one-step control input constraint representation:
% %     Gu_ext * z_ext <= gu_ext

% % Np-horizon state constraint representation:
% %     blkdiag(Gu_ext, ..., Gu_ext) * Z_ext <= [gu_ext; ...; gu_ext]

% % --> blkdiag(Gu_ext, ..., Gu_ext) * B_bar*dU <= [gu_ext; ...; gu_ext]
% %     - blkdiag(Gu_ext, ..., Gu_ext) * A_bar*z_ext(0)

% % (c) On control rate constraints:

% % One-step control rate constraint representation:
% %     Kdu * du <= kdu

% % Np-horizon state constraint representation:
% %     blkdiag(Kdu, ..., Kdu) * dU <= [kdu; ...; kdu]

% % In total, we obtain all constraints on decision variable dU as:
% % [blkdiag(Fz_ext, ..., Fz_ext) * B_bar;  |         [[fz_ext; ...; fz_ext]; |     [ blkdiag(Fz_ext, ..., Fz_ext) * A_bar; |
% % |blkdiag(Gu_ext, ..., Gu_ext) * B_bar;  | * dU <= |[gu_ext; ...; gu_ext]; | -   | blkdiag(Gu_ext, ..., Gu_ext) * A_bar; |
% % |blkdiag(Kdu, ..., Kdu)                 ]         |[kdu; ...; kdu]        ]     | 0                                     ]

% % The following code lines realize the above expression:

[A_bar,B_bar] = genConMat(A,B,Np);

Uad.A = [kron(eye(Np),Fz_ext)*B_bar;
         kron(eye(Np),Gu_ext)*B_bar;
         kron(eye(Np),Kdu)];
Uad.b = [kron(ones(Np,1),fz_ext);
         kron(ones(Np,1),gu_ext);
         kron(ones(Np,1),kdu)];
Uad.B = [kron(eye(Np),Fz_ext)*A_bar;
         kron(eye(Np),Gu_ext)*A_bar;
         zeros(size(kron(ones(Np,1),kdu),1), size(A_bar,2))];
Uad.Ae = zeros(0,size(B_bar,2));
Uad.be = zeros(0,1);
end
 
function [x_hat, d_hat] = KalmanDisObs(x_hat,d_hat,u,y,Ad,Bd,Cd,L1,L2)
% % This function realize Observer 1

x_hat = Ad*x_hat + Bd*u + Bd*d_hat + L1*(y-Cd*x_hat);
d_hat = d_hat + L2*(y-Cd*x_hat);
end

function [p,d_hat] = ssObserver(L,p,d_hat,x,u,A,Bd)
% % This function realize Observer 2

%{
Reference:
Kim, Kyung-Soo, and Keun-Ho Rew. "Reduced order disturbance observer for 
discrete-time linear systems." Automatica 49.4 (2013): 968-975.
%}

p = p - L*((A-eye(size(A)))*x + Bd*u + Bd*d_hat);
d_hat = p + L*x;
end