clear; clc;

% ========= Colors =========
col200 = [0.20 0.45 0.75];   % TSP200
col500 = [0.75 0.15 0.15];   % TSP500

% ========= Load TSP200 =========
data200 = readtable('wandb_export_2025-08-29T11_40_53.337+07_00.csv','VariableNamingRule','preserve');
x200 = double(data200.('Step'));
y200 = double(data200.('[ppo_faco]tsp200_sd0 - train_mean_cost'));

% ========= Load TSP500 =========
data500 = readtable('wandb_export_2025-08-29T11_40_39.103+07_00.csv','VariableNamingRule','preserve');
x500 = double(data500.('Step'));
y500 = double(data500.('[ppo_faco]tsp500_sd0 - train_mean_cost'));

% ---- Clean + sort ----
m200 = isfinite(x200) & isfinite(y200);  
x200 = x200(m200);  y200 = y200(m200);
[x200, i200] = sort(x200);  y200 = y200(i200);

m500 = isfinite(x500) & isfinite(y500);  
x500 = x500(m500);  y500 = y500(m500);
[x500, i500] = sort(x500);  y500 = y500(i500);

% ========= One figure, TWO subplots (RAW) =========
fig = figure('Color','w');
t = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');

% --- Left: TSP200 (raw) ---
ax1 = nexttile(t); hold(ax1,'on');
plot(ax1, x200, y200, 'LineWidth',1.8, 'Color',col200);
grid(ax1,'on'); box(ax1,'on');
set(ax1,'TickLabelInterpreter','latex','FontSize',15, ...
    'GridColor',[0.70 0.70 0.70],'GridAlpha',0.55,'LineWidth',1.2);
xlabel(ax1,'Step','Interpreter','latex','FontSize',18);
ylabel(ax1,'Sampling Cost','Interpreter','latex','FontSize',18);
title(ax1,'TSP200','Interpreter','latex','FontSize',18);

% --- Right: TSP500 (raw) ---
ax2 = nexttile(t); hold(ax2,'on');
plot(ax2, x500, y500, 'LineWidth',1.8, 'Color',col500);
grid(ax2,'on'); box(ax2,'on');
set(ax2,'TickLabelInterpreter','latex','FontSize',15, ...
    'GridColor',[0.70 0.70 0.70],'GridAlpha',0.55,'LineWidth',1.2);
xlabel(ax2,'Step','Interpreter','latex','FontSize',18);
ylabel(ax2,'Sampling Cost','Interpreter','latex','FontSize',18);
title(ax2,'TSP500','Interpreter','latex','FontSize',18);

% Save
set(fig,'Units','inches','Position',[1 1 10 4]); 
drawnow;
exportgraphics(t, 'TSP200_TSP500_Test.pdf', ...
    'ContentType','vector', ...
    'BackgroundColor','w');
