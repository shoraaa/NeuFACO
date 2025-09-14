clear; clc;

% Load TSP200
data200 = readtable('wandb_export_2025-08-29T11_40_53.337+07_00.csv', ...
    'VariableNamingRule', 'preserve');
x200 = double(data200.('Step'));
y200 = double(data200.('[ppo_faco]tsp200_sd0 - train_mean_cost'));

% Load TSP500
data500 = readtable('wandb_export_2025-08-29T11_40_39.103+07_00.csv', ...
    'VariableNamingRule', 'preserve');
x500 = double(data500.('Step'));
y500 = double(data500.('[ppo_faco]tsp500_sd0 - train_mean_cost'));

% ==== Vẽ 2 graph cạnh nhau ====
figure;
tiledlayout(1,2,"Padding","compact","TileSpacing","compact");

% --- Graph TSP200 ---
nexttile; hold on;
plot(x200, y200, 'Color',[0.1 0.3 0.7],'LineWidth',1.2,'DisplayName','TSP200 Raw');
grid on; box on;
set(gca,'TickLabelInterpreter','latex','FontSize',10);
xlabel('Step','Interpreter','latex','FontSize',12);
ylabel('Sampling Cost','Interpreter','latex','FontSize',12);
legend('Location','northeast','Interpreter','latex');
title('TSP200','Interpreter','latex','FontSize',12);

% --- Graph TSP500 ---
nexttile; hold on;
plot(x500, y500, 'Color',[0.9 0.4 0.1],'LineWidth',1.2,'DisplayName','TSP500 Raw');
grid on; box on;
set(gca,'TickLabelInterpreter','latex','FontSize',10);
xlabel('Step','Interpreter','latex','FontSize',12);
ylabel('Sampling Cost','Interpreter','latex','FontSize',12);
legend('Location','northeast','Interpreter','latex');
title('TSP500','Interpreter','latex','FontSize',12);

% ==== Auto save ====
set(gcf,'Units','inches','Position',[1 1 10 4]);  % fix size 10x4 inch
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 4]);

print(gcf,'TSP200_vs_TSP500_Tes.png','-dpng','-r300');   % lưu PNG 300dpi
