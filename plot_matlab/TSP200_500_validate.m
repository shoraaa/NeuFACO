clear; clc;

% ==== Load TSP200 ====
data200 = readtable('wandb_export_2025-07-31T13_57_17.936+07_00.csv', ...
    'VariableNamingRule', 'preserve');
x200 = double(data200.('Step'));
y200 = double(data200.('[ppo_faco]tsp200_sd0 - val_best_aco_T'));
y200_gauss = smoothdata(y200, 'gaussian', 50);

% ==== Load TSP500 ====
data500 = readtable('wandb_export_2025-07-31T13_57_11.098+07_00.csv', ...
    'VariableNamingRule', 'preserve');
x500 = double(data500.('Step'));
y500 = double(data500.('[ppo_faco]tsp500_sd0 - val_best_aco_T'));
y500_gauss = smoothdata(y500, 'gaussian', 50);

% ==== Vẽ 2 graph cạnh nhau (200 left, 500 right) ====
figure;
tiledlayout(1,2,"Padding","compact","TileSpacing","compact");

% --- Graph TSP200 (left) ---
nexttile; hold on;
plot(x200, y200, 'Color',[0.6 0.6 0.6 0.4],'LineWidth',0.8,'DisplayName','Raw');
plot(x200, y200_gauss, 'Color',[0.1 0.3 0.7],'LineWidth',1.8,'DisplayName','Gaussian Smooth');
grid on; box on;
set(gca,'TickLabelInterpreter','latex','FontSize',10);
xlabel('Step','Interpreter','latex','FontSize',12);
ylabel('Sampling Cost','Interpreter','latex','FontSize',12);
title('TSP200','Interpreter','latex','FontSize',12);
legend('Location','northeast','Interpreter','latex');

% --- Graph TSP500 (right) ---
nexttile; hold on;
plot(x500, y500, 'Color',[0.6 0.6 0.6 0.4],'LineWidth',0.8,'DisplayName','Raw');
plot(x500, y500_gauss, 'Color',[0.9 0.4 0.1],'LineWidth',1.8,'DisplayName','Gaussian Smooth');
grid on; box on;
set(gca,'TickLabelInterpreter','latex','FontSize',10);
xlabel('Step','Interpreter','latex','FontSize',12);
ylabel('Sampling Cost','Interpreter','latex','FontSize',12);
title('TSP500','Interpreter','latex','FontSize',12);
legend('Location','northeast','Interpreter','latex');

% ==== Auto save ====
set(gcf,'Units','inches','Position',[1 1 10 4]); % size 10x4 inch
print(gcf,'TSP200_left_TSP500_right.png','-dpng','-r300');
