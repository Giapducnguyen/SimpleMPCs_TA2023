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
Ts = 0.1;                       % sampling period
sysd = c2d(ss(Ac,Bc,Cc,Dc),Ts); % discretization
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Extended-state dynamics

% % Extend the system state to include disturbance estimate w_hat(k)
% % z(k) = [x(k); w_hat(k)]
% % z(k+1) = Az * z(k) + Bz * u(k)
% % y(k) = Cd*x(k) = Cz * z(k)

Az = [Ad,                       Bd;
      zeros(nd,size(Ad,2)),     eye(nd)];
Bz = [Bd; zeros(nd,size(Bd,2))];
Cz = [Cd, zeros(size(Cd,1),nd)];

nz = size(Az,2); % number of extended state z

%% State constraints
x1_min = -1;    x1_max = 1.1;
x2_min = -0.2;  x2_max = 0.2;

%% Control constraints
u_min = -0.1;       u_max = 0.1;        % limitations on magnitude
du_min = -0.05;     du_max = 0.05;      % limitations on rate of change

%% MPC data
Np = 15;            % Prediction horizon
R = 1;              % Weighting matrices on control rate
Qe = 1;             % Weighting matrices on tracking error

%% MPC Controller Object 

model = LTISystem('A',Az,'B',Bz,'C',Cz,'D',zeros(ny,nu),'Ts',Ts);

model.x.min = [x1_min; x2_min; -2];
model.x.max = [x1_max; x2_max;  2];
model.u.min = u_min;
model.u.max = u_max;
model.u.with('deltaMin');
model.u.with('deltaMax'); 
model.u.deltaMin = du_min;
model.u.deltaMax = du_max;

model.y.with('reference');
model.y.reference = 'free';
model.y.penalty = QuadFunction(Qe);
model.u.with('deltaPenalty'); 
model.u.deltaPenalty = QuadFunction(R);

mpc = MPCController(model,Np);

% ectrl = mpc.toExplicit();

%% Observer Gains Calculation

% % This observer requires STATE-feedback, see more at the Function Helper
% % section (end of this file)
% % Observer gains selection:

% L = 0.5*[0, 5];                  % Tuning Option 1: by Trial-and-error
L = place(eye(nx),Bd, [0.75, 1]); % tuning Option 2: from stability analysis

%% Main Simulation Loop
Nsim = 400;     % Number of simulation steps

% % Initialization
infeaOCP = 0;      % Infeasible OCP counter
x0 = [0;0];        % Initial state
u0 = 0;             % Initial control             
r0 = 1;            % Unit reference

p = 0;      w_h = 0;         % initialization for Disturbance Observer

% % State logging
X_log = zeros(nx,Nsim);             % original state logging
X_log(:,1) = x0;

x = x0;                             % temporary state

% % Control logging
U_log = zeros(nu,Nsim);
U_log(:,1) = u0;

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
    elseif (i-1)*Ts < 12
        r = r0;
        w = 0;
    elseif (i-1)*Ts < 25
        r = 0;
    else
        r = 0;
        w = 0.07;
    end

    % % Observation
    [p,w_h] = ssObserver(L,p,w_h,x,u0,Ad,Bd);

    % % Calling solver and solve for optimal control increment du*
    tic;
    % Implicit
    [Umpc, feasible, openloop] = mpc.evaluate([x;w_h],'u.previous',u0,'y.reference',r);
    % Explicit
    % [Umpc, feasible, openloop] = ectrl.evaluate([x;w_h],'u.previous',u0,'y.reference',r);
    ExecutionTime(i) = toc; % execution time logging

    % % If solver return infeasible OCP, apply zero move: du* = 0 and
    % % increase counter by 1, else, take the optimal control Umpc
    if feasible ~= 1
        u_c = 0; infeaOCP = infeaOCP + 1;
    else
        u_c = Umpc;
    end
    
    % % Loggings
    U_log(:,i) = u_c;           % control
    dU_log(:,i) = u_c-u0;       % control rate
    R_log(:,i) = r;             % reference
    W_log(:,i) = w;             % disturbance
    What_log(:,i) = w_h;        % disturbance estimation
    X_log(:,i) = x;             % state

    % % Propagate system forward
    x = Ad*x + Bd*(u_c + w);
    
    % % Assign current control move for the next loop
    u0 = u_c;
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

function [p,d_hat] = ssObserver(L,p,d_hat,x,u,Ad,Bd)
% % This function realize Observer 2

%{
Reference:
Kim, Kyung-Soo, and Keun-Ho Rew. "Reduced order disturbance observer for 
discrete-time linear systems." Automatica 49.4 (2013): 968-975.
%}

p = p - L*((Ad-eye(size(Ad)))*x + Bd*u+Bd*d_hat);
d_hat = p + L*x;
end