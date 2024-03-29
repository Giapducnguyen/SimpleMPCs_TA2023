% % Robust Tube MPC
close all; clear all; clc

%% System Data
A = [1 1; 0 1];
B = [0.5; 1];

%% Original State Constraints
x1_min = -10;               x2_min = -2;
x1_max = 2;                 x2_max = 2;
x_min = [x1_min; x2_min];   x_max = [x1_max; x2_max];

%% Original Control Input Constraints
u_min = -1;                 u_max = 1;

%% Disturbance Limitations
w_min = [-0.15; -0.15];     w_max = -w_min;

%% MPC Data
Q = eye(2);     % State penalty matrix
R = 1;          % Control penalty matrix
Np = 12;        % Prediction horizon

%% Disturbance Invariant Set Calculation

% % In this section, we will calculate tightened sets for state and control
% % input, considering all possible disturbance values (worst-case scenario)

% % We first represent state, control and disturbance sets in the
% % H-representation format (Polyhedral format). This step requires MPT3
% % toolbox.

set_x = Polyhedron('lb', x_min, 'ub', x_max); % State
set_u = Polyhedron('lb', u_min, 'ub', u_max); % Control
set_w = Polyhedron('lb', w_min, 'ub', w_max); % Disturbance

% % Second, we calculate the disturbance invariant set of state error (
% % state error = actual state - nominal state ). This is officially named
% % Robust Positively Invariant (RPI) set.

% % Given the actual control policy u_act = u_mpc + K (x_act-x_nom)
% % We first find the feedback control gain K and the corresponding
% % closed-loop state matrix A_cl = A-B*K

% % The feedback control gain K can be found by solving the discrete-time
% % Riccati Equation by function dlqr: 
[K,P,~] = dlqr(A,B,Q,R);

% % (Or using function idare)
% [P,K,~] = idare(A,B,Q,R,[],[]);

% % Note: matrix P can be used as a terminal penalty for mpc formulation

% % Closed-loop state matrix A_cl:
A_cl = A - B*K;

% % Disturbance Invariant Set

% % Option 1: Guessing method
% % Note: + means Minkowski sum; - means Pontryagin difference; 
% %       * means affine mapping in this scenario (MPT3 toolbox required).

% figure(1); hold on;
% set_e = set_w;
% k_max = 5;
% for k = 1 : k_max
%     set_e = set_e + (A_cl^k)*set_w;
%     set_e.plot('Color','r','alpha',0.1); pause(0.5);
% end
% title('Disturbance invariant set (RPI set)','interpreter','latex','Fontsize',12)
% figure(1); hold off;
% set_e.minVRep(); set_e.minHRep(); % remove redundant expressions

% % Option 2: Outer (epsilon-) approximation
% % Reference:
% % Rakovic, Sasa V., et al. "Invariant approximations of the minimal 
% % robust positively invariant set." IEEE Transactions on automatic 
% % control 50.3 (2005): 406-410.

epsilon = 1e-5;
set_e = eApprox(epsilon,A_cl,set_w);
set_e.minVRep(); set_e.minHRep(); % remove redundant expressions

%% Constraint Tightening
% % Based on the RPI set calulated above, we proceed to tighten the
% % original state and control set to account for the worst-case scenario.
% % As long as the nominal state and control inside these sets, the actual
% % state and control will naver violate the given original constraints.

% % State constraint tightening
set_xbar = set_x - set_e;

% % Control constraint tightening
set_ubar = set_u - (-K*set_e);

% % Plots for illustration

% % State constraints
figure(3); hold on;
px1 = set_x.plot('color','m','alpha',0.2);
px2 = set_xbar.plot('color','r','alpha',0.2);
legend([px1 px2],{'original','tightened'},'interpreter','latex','Fontsize',14)
title('State Constraint Set','interpreter','latex','Fontsize',12)
hold off;

% % Control constraints
figure(4); hold on;
pu1 = set_u.plot('color','r');
pu2 = set_ubar.plot('color','g');
legend([pu1 pu2],{'original','tightened'},'interpreter','latex','Fontsize',14)
title('Control constraint set','interpreter','latex','Fontsize',12)
hold off;

%% Terminal Set Computation
sys = LTISystem('A',A,'B',B);
set_xfbar = sys.invariantSet('X',set_xbar,'U',set_ubar);

%% Initialization

% Simulation horizon
Nsim = 12;

% % Initial state
x0 = [-7.5; -1.6];

% % Nominal state initilization
X_nom = zeros(2,Nsim+1);
X_nom(:,1) = x0;

% % Actual state initialization
X_act = zeros(2,Nsim+1);
X_act(:,1) = [-7.5; -1.8]; % this can be different from x0

% % Execution time initialization
Exe_time = nan(1, Nsim);

% % Actual control initialization
U_act = nan(1,Nsim);

% % Nominal control initialization
U_nom = nan(1,Nsim);

% % Disturbance initialization
W = nan(2,Nsim);

% % Note: For simple Tube MPC, there are two methods of implementation.

% % Method 1: Open-loop roll-out policy
% % Solve MPC problem offline to obtain an optimal control sequence then
% % at each time instant, extract the corresponding control move

% % Method 2: Solve MPC problem online as usual.

% % Select mpcMethod = 1 or 2
mpcMethod = 1; % offline; 2 - online

% % Offline Computation
U_rollOut = mpcPolicy(X_nom(:,1),A,B,Q,P,R,Np,set_xbar,set_ubar,set_xfbar);

%% Main Simulation Loop

for k = 1:Nsim

    % mpc policy computation
    tic;
    if mpcMethod == 1           % offline
        u_mpc = U_rollOut(k,:);
    else                        % online
        U_online = mpcPolicy(X_nom(:,k),A,B,Q,P,R,Np,set_xbar,set_ubar,set_xfbar);
        u_mpc = U_online(1,:);
    end
    Exe_time(k) = toc;          % execution time logging

    % Propagate prediction model forward
    X_nom(:,k+1) = A*X_nom(:,k) + B*u_mpc;

    % Compute actual control
    u_act = u_mpc + (-K*( X_act(:,k)-X_nom(:,k) ) );
    
    % Random disturbance generation
    w = w_min + rand(2,1).*(w_max-w_min);
    % or showcasing purpose only
    w = [-0.15; 0.15];

    % Apply actual control to the system
    X_act(:,k+1) = A*X_act(:,k) + B*u_act + w;
    
    % Loggings
    U_act(k) = u_act;       % actual control
    U_nom(k) = u_mpc;       % nominal control
    W(:,k) = w;               % disturbance
end

%% Plotting
% % This is the position of the RPI set corresponding to the nominal state
% % trajectory calculated above.

% % (a) RPI set plot

figure(5); hold on;
px1 = set_x.plot('color','m','alpha',0.2);
px2 = set_xbar.plot('color','r','alpha',0.2);

for i = 1: Nsim+1
    RPI_Tube = set_e + X_nom(:,i);
    px3 = RPI_Tube.plot('color','g','alpha',0.5);
end

% % (b) state space plot
figure(5);
px4 = plot(X_nom(1,:),X_nom(2,:),'r','LineWidth',1.5,'Marker','s','MarkerSize',6);
px5 = plot(X_act(1,:),X_act(2,:),'b','LineWidth',1.5,'Marker','*','MarkerSize',6);
xlabel('$x_1$','interpreter','latex','Fontsize',12)
ylabel('$x_2$','interpreter','latex','Fontsize',12)
legend([px1 px2 px3 px4 px5],{'original','tightened','RPI','predicted','actual'},...
    'interpreter','latex','Fontsize',12);
title('System State Trajectories','interpreter','latex','Fontsize',12)
xlim([-11 3]); ylim([-2.4 2.4]);
box on; grid on;

figure(6); hold on;
stairs(1:Nsim, U_act,'b-','LineWidth',1.5)
stairs(1:Nsim, U_nom,'r-.','LineWidth',1.5)
stairs(1:Nsim, u_max*ones(1,Nsim),'k--','LineWidth',1.5)
stairs(1:Nsim, max(set_ubar.V)*ones(1,Nsim),'g--','LineWidth',1.5)
stairs(1:Nsim, u_min*ones(1,Nsim),'k--','LineWidth',1.5)
stairs(1:Nsim, min(set_ubar.V)*ones(1,Nsim),'g--','LineWidth',1.5)
legend({'actual control','nominal control','original constraints','tightened constraints'},'interpreter','latex','Fontsize',12)
title('Control Trajectories','interpreter','latex','Fontsize',12)
ylim([-1.2 1.2])
box on; grid on;
hold off;


%% Function Helper

% % epsilon-based outer approximation of RPI set
function Fs_alpha = eApprox(epsilon,A_cl,set_w)
[nx,~] = size(A_cl);
Ms = 1000;
s = 0;
alp = 1000;

mss = zeros(2*nx,1);

while(alp > epsilon/(epsilon+Ms))
    s = s+1;
    alp = max(set_w.support(A_cl^s*set_w.A')./set_w.b);
    mss = mss + set_w.support([A_cl^s, -A_cl^s]);
    Ms = max(mss);
end

figure(2); hold on;
Fs = set_w;
for i = 1:s-1
    Fs = Fs + A_cl^i*set_w;
    Fs.plot('Color','red','alpha',0.1); pause(0.05);
end
Fs_alpha = ((1-alp)^-1)*Fs;
Fs_alpha.plot('Color','yellow','alpha',0.35);
title('Disturbance invariant set (RPI set)','interpreter','latex','Fontsize',12)
figure(2); hold off;

end

% % Lifted Model Matrices
function [B_bar,A_bar] = liftedModel(A,B,Np)
B_bar = tril(cell2mat(cellfun(@(x)A^x,num2cell(toeplitz(0:Np-1)),'UniformOutput',0)))*kron(eye(Np),B);
A_bar = cell2mat(cellfun(@(x)A^x,num2cell((1:Np)'),'UniformOutput',0));
end

% % Admissible Inputs
function Uad = admissibleInputs(A,B,set_xbar,set_ubar,set_xfbar,Np)
n = size(A,1);
[B_bar,A_bar] = liftedModel(A,B,Np);

Uad.A = [kron(eye(Np),set_ubar.A);blkdiag(kron(eye(Np-1),set_xbar.A),set_xfbar.A)*B_bar];
Uad.b = [kron(ones(Np,1),set_ubar.b);[kron(ones(Np-1,1),set_xbar.b);set_xfbar.b]];
Uad.B = [zeros(Np*size(set_ubar.b,1),n) ; blkdiag(kron(eye(Np-1),set_xbar.A),set_xfbar.A)*A_bar];
Uad.Ae = zeros(0,size(B_bar,2));
Uad.be = zeros(0,1);
end

% % MPC policy
function Vmpc = mpcPolicy(x0,A,B,Q,P,R,Np,set_xbar,set_ubar,set_xfbar)

Uad = admissibleInputs(A,B,set_xbar,set_ubar,set_xfbar,Np);

[S,M] = liftedModel(A,B,Np);
Q_bar = blkdiag(kron(eye(Np-1),Q),P);
R_bar = kron(eye(Np),R);
H = S'*Q_bar*S + R_bar; 
q = S'*Q_bar*M*x0;

% Vmpc = quadprog(H,q,Uad.A,Uad.b-Uad.B*x0); % slower

options = mpcActiveSetOptions;
iA0 = false(size(Uad.b));
Vmpc = mpcActiveSetSolver(H,q,Uad.A,Uad.b-Uad.B*x0,Uad.Ae,Uad.be,iA0,options); % faster
end