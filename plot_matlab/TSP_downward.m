%% ================== CLEAN START ==================
clear; clc; close all;

% Nền trắng và chữ đen
set(groot,'defaultFigureColor','w');
set(groot,'defaultAxesColor','w');
set(groot,'defaultAxesXColor','k');
set(groot,'defaultAxesYColor','k');
set(groot,'defaultAxesFontSize',18);  % tick số to hơn
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% ================== FILE LIST ==================
% DeepACO, GFACS, PPO-FACO
files200 = { ...
'test_result_ckptdeepaco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0_iterations.csv', ...
'test_result_ckptgfacs_200-tsp200-ninst64-AS-nants100-niter100-nruns1-seed0_iterations.csv', ...
'test_result_ckptppo_faco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0_iterations.csv'};

files500 = { ...
'test_result_ckptdeepaco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0_iterations.csv', ...
'test_result_ckptgfacs_500-tsp500-ninst64-AS-nants100-niter100-nruns1-seed0_iterations.csv', ...
'test_result_ckptppo_faco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0_iterations.csv'};

labels = {'DeepACO','GFACS','NeuFACO'};
colors = [0.75 0.15 0.15;   % DeepACO - đỏ
          0.20 0.45 0.75;   % GFACS   - xanh dương
          0.90 0.55 0.20];  % PPO-FACO- cam

%% ================== DRAW ==================
figure('Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% ---------- TSP200 ----------
nexttile; hold on;
for i = 1:numel(files200)
    T = readtable(files200{i},'VariableNamingRule','preserve');
    [x,y] = chooseXY(T);
    plot(x, y, 'LineWidth', 2.0, ...
        'DisplayName', labels{i}, 'Color', colors(i,:));
end
xlabel('Iterations','Color','k','FontSize',22,'FontWeight','bold');
ylabel('Cost','Color','k','FontSize',22,'FontWeight','bold');
title('TSP200','Color','k','FontSize',22,'FontWeight','bold');
legend('show','Location','best','FontSize',22);
box on; grid on; 
set(gca,'GridColor',[0.70 0.70 0.70],'GridAlpha',0.55, ...
        'LineWidth',1.2);  % trục dày hơn

% ---------- TSP500 ----------
nexttile; hold on;
for i = 1:numel(files500)
    T = readtable(files500{i},'VariableNamingRule','preserve');
    [x,y] = chooseXY(T);
    plot(x, y, 'LineWidth', 2.0, ...
        'DisplayName', labels{i}, 'Color', colors(i,:));
end
xlabel('Iterations','Color','k','FontSize',22,'FontWeight','bold');
ylabel('Cost','Color','k','FontSize',22,'FontWeight','bold');
title('TSP500','Color','k','FontSize',22,'FontWeight','bold');
legend('show','Location','best','FontSize',22);
box on; grid on; 
set(gca,'GridColor',[0.70 0.70 0.70],'GridAlpha',0.55, ...
        'LineWidth',1.2);

%% ================== AUTO SAVE ==================
set(gcf,'Units','inches','Position',[1 1 12 5]);

% Export current figure as PDF (vector format)
exportgraphics(gcf, 'TSP200_vs_TSP500_3lines.pdf', ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'w');


%% ================== LOCAL FUNCTION ==================
function [x,y] = chooseXY(T)
% Tự động chọn cột iter & best. Nếu không có thì lấy cột 1 & 2.
    vars = T.Properties.VariableNames;
    idIter = find(strcmpi(vars,'iter'), 1);
    idBest = find(strcmpi(vars,'best'), 1);

    if ~isempty(idIter), x = T.(vars{idIter}); else, x = T.(vars{1}); end
    if ~isempty(idBest)
        y = T.(vars{idBest});
    else
        y = T.(vars{2});
    end

    x = double(x); y = double(y);
end
