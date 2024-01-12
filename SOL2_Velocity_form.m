% % Double integrator problem - with disturbance
% % Code for the Reference Tracking formulation 4 of
% % Module 5 - Introduction to MPC - Kolmanovsky
clear all; close all; clc

%% Continuous-time model
Ac = [0,1;0,0];
Bc = [0;1];
Cc = [1,0];
Dc = 0;

nx = size(Ac,2);    % number of states
nu = size(Bc,2);    % number of inputs
ny = size(Cc,1);    % number of outputs
nr = 1;             % tracking on x1 only
nd = 1;             % number of disturbances

%% Discrete-time model
Ts = 0.1;                           % sampling period
sysd = c2d(ss(Ac,Bc,Cc,Dc),Ts);     % discretization
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Extend state matrices - x_{ext,k} = [Delta x_{k}; e; x_{k-1}; u_{k-1}]
A_ext = [Ad,            zeros(nx,nr),   zeros(nx,nx),   zeros(nx,nu);
         Cd*Ad,         eye(nr,nr),     zeros(nr,nx),   zeros(nr,nu);
         eye(nx,nx),    zeros(nx,nr),   eye(nx,nx),     zeros(nx,nu);
         zeros(nu,nx),  zeros(nu,nr),   zeros(nu,nx),   eye(nu,nu)];

B_ext = [Bd;
         Cd*Bd;
         zeros(nx,nu);
         eye(nu)];

%% State constraints
x1_min = -1;    x1_max = 1.1;
x2_min = -0.2;  x2_max = 0.2;

% H-representation of state constraint:       Fx * x <= fx
Fx = [1, 0;
     -1, 0;
      0  1;
      0 -1]; 
fx = [x1_max;-x1_min;x2_max;-x2_min];

% Extend to extended state:     Fz * x_ext <= fz
% x = [0 0 1 0]*x_ext  --> Fx * ([0 0 1 0]*x_ext) <= fx
% --> Fz = Fx * [0 0 1 0]  &  fz = fx
Fz = Fx*[zeros(nx), zeros(nx,nr), eye(nx), zeros(nx,nu)];
fz = fx;

%% Control constraints
u_min = -0.1; u_max = 0.1;
du_min = -0.05;     du_max = 0.05;      % limitations on rate of change

% H-representation of input constraints:       Gu * u <= gu
Gu = [1;-1]; 
gu = [u_max;-u_min];

% Extend to extended state: Gz * x_ext <= gz
% u = [0 0 0 1]*x_ext --> Gu * ([0 0 0 1]*x_ext) <= gu 
% --> Gz = Gu * [0 0 0 1] && gz = gu
Gz = Gu*[zeros(nu,nx), zeros(nu,nr), zeros(nu,nx), eye(nu)];
gz = gu;

Kdu = [1; -1];
kdu = [du_max; -du_min];

%% MPC data
Np = 8;
R = 1; Qe = 1;
Q = blkdiag(zeros(nx), Qe*eye(nr), zeros(nx), zeros(nu));

%% Terminal penalty calculation for [Delta x; e]
A_pat = [Ad,    zeros(nx,nr);
         Cd*Ad, eye(nr)];
B_pat = [Bd;
         Cd*Bd];

Q_pat = blkdiag(zeros(nx), Qe);
R_pat = R;

[P,~,~,~] = idare(A_pat,B_pat,Q_pat,R_pat,[],[]);
Qf = blkdiag(P,zeros(nx),zeros(nu));

%% Stacked constraints
Uad = admissibleInputs(A_ext,B_ext,Np,Fz,fz,Gz,gz, Kdu, kdu);

%% Quadratic Programming
Q_bar = blkdiag(kron(eye(Np-1),Q),Qf);
R_bar = kron(eye(Np),R);
[A_bar,B_bar] = genConMat(A_ext,B_ext,Np);
H = B_bar'*Q_bar*B_bar + R_bar;
options = mpcActiveSetOptions;
iA0 = false(size(Uad.b));

%% Simulation loop
Nsim = 350;
infea = 0;

% % Initialization
x0 = [0;0]; 
u0 = 0;
r0 = 1;

X_log = zeros(nx,Nsim+1);
X_log(:,1) = x0;

Xext_log = zeros(nx+nr+nx+nu,Nsim);
x = x0;

U_log = zeros(nu,Nsim);
U_log(:,1) = u0;

% % Control rate logging
dU_log = zeros(nu,Nsim);

R_log = zeros(nr,Nsim);
R_log(:,1) = r0;

W_log = zeros(1,Nsim);

ExecutionTime = nan(1,Nsim);

for k = 1:Nsim
    % Disturbance & reference assignment
    if (k-1)*Ts < 1
        r = 0;
        w = 0;
    elseif (k-1)*Ts < 10
        r = r0;
        w = 0;
    elseif (k-1)*Ts < 25
        r = 0;
    else
        r = 0;
        w = 0.07;
    end
    % current, previous state control assignment 
    if k == 1
        x_c = x0; x_p = x0; u_p = u0;
    else 
        x_c = X_log(:,k); x_p = X_log(:,k-1); u_p = U_log(:,k-1);
    end
    % Calculate the extended state
    Xext_log(:,k) = [x_c-x_p; Cd*x_c-r; x_p; u_p];
    % Caluclate linear cost of QP
    q = B_bar'*Q_bar*A_bar*Xext_log(:,k);
    % Solve for optimal control
    tic;
    [dU,exitflag,iA0,~] = mpcActiveSetSolver(H,q,Uad.A,Uad.b-Uad.B*Xext_log(:,k),Uad.Ae,Uad.be,iA0,options);
%     [dU,~,exitflag,~] = quadprog(H,q,Uad.A,Uad.b-Uad.B*Xext_log(:,k),Uad.Ae,Uad.be,[],[],double(iA0),options);
    ExecutionTime(k) = toc;
    % calling backup controller if the optimization problem is infeasible
    if exitflag <= 0
        du = 0; infea = infea + 1;
    else
        du = dU(1,nu);
    end
    % data loggings
    u_c = u_p + du;
    U_log(:,k) = u_c;
    dU_log(:,k) = du;           % control rate
    R_log(:,k) = r;
    W_log(:,k) = w;
    X_log(:,k) = x_c;
    % update state
    X_log(:,k+1) = Ad*X_log(:,k) + Bd*(U_log(:,k) + W_log(:,k)); % add noise: + 0.01*randn
end

%% Plotting
lw = 1.5;
TimeArray = 0:Ts:(Ts*Nsim);

figure(1);
fig1 = tiledlayout(5,1);

nexttile
plot(TimeArray(1,1:end-1),X_log(1,1:end-1),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),R_log(1,:),'LineStyle','--','Color','b','LineWidth',lw);
plot(TimeArray(1,1:end-1),x1_max*ones(size(X_log(1,1:end-1))),'k--','LineWidth',lw)
ylabel('State $x_1$','interpreter','latex')
legend({'$x_1$','reference','constraints'},'interpreter','latex')
ylim([-0.1 1.2])
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray(1,1:end-1),X_log(2,1:end-1),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),x2_max*ones(size(X_log(2,1:end-1))),'k--','LineWidth',lw);
plot(TimeArray(1,1:end-1),x2_min*ones(size(X_log(2,1:end-1))),'k--','LineWidth',lw);
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
plot(TimeArray(1,1:end-1),W_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('Disturbance $w$','interpreter','latex')
% xlabel('Time (seconds)','interpreter','latex')
legend({'$w$'},'interpreter','latex')
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


%% Function helper

function Uad = admissibleInputs(A,B,Np,Fz,fz,Gz,gz,Kdu,kdu)

[A_bar,B_bar] = genConMat(A,B,Np);

Uad.A = [kron(eye(Np),Fz)*B_bar;
         kron(eye(Np),Gz)*B_bar;
         kron(eye(Np),Kdu)];
Uad.b = [kron(ones(Np,1),fz);
         kron(ones(Np,1),gz);
         kron(ones(Np,1),kdu)];
Uad.B = [kron(eye(Np),Fz)*A_bar;
         kron(eye(Np),Gz)*A_bar;
         zeros(size(kron(ones(Np,1),kdu),1), size(A_bar,2))];
Uad.Ae = zeros(0,size(B_bar,2));
Uad.be = zeros(0,1);
end

function [A_bar,B_bar] = genConMat(A,B,Np)

% A_bar = cell2mat(cellfun(@(x)A^x,num2cell((1:Np)'),'UniformOutput',false));
% B_bar = tril(cell2mat(cellfun(@(x)A^x,num2cell(toeplitz(0:Np-1)),'UniformOutput',false)))*kron(eye(Np),B);

% % or using cell

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

end
