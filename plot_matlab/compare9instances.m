clear; clc;

% ==== Danh sách file CSV và instance ====
files = { ...
    'wandb_export_2025-08-29T10_44_37.219+07_00.csv', ... % d493
    'wandb_export_2025-08-29T10_44_34.420+07_00.csv', ... % d657
    'wandb_export_2025-08-29T10_44_30.507+07_00.csv', ... % fl417
    'wandb_export_2025-08-29T10_44_26.263+07_00.csv', ... % lin318
    'wandb_export_2025-08-29T10_44_20.895+07_00.csv', ... % p654
    'wandb_export_2025-08-29T10_44_12.366+07_00.csv', ... % pcb442
    'wandb_export_2025-08-29T10_44_02.806+07_00.csv', ... % pr439
    'wandb_export_2025-08-29T10_44_17.260+07_00.csv', ... % rat575
    'wandb_export_2025-08-29T10_44_08.574+07_00.csv'  ... % rd400
};

instances = {'d493','d657','fl417','lin318','p654','pcb442','pr439','rat575','rd400'};

% ==== Tạo figure 3x3 ====
figure;
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for i = 1:numel(files)
    % Đọc file
    data = readtable(files{i}, 'VariableNamingRule','preserve');
    x = double(data.('Step'));
    inst = instances{i};

    % Lấy raw cost
    y_deepaco = double(data.(sprintf("deepaco-500nodes-128ants-50iter-seed0 - cost/['%s']", inst)));
    y_gfacs   = double(data.(sprintf("gfacs_500-500nodes-128ants-50iter-seed0 - cost/['%s']", inst)));
    y_ppo     = double(data.(sprintf("ppo_faco_500-500nodes-128ants-50iter-seed0 - cost/['%s']", inst)));

    % Vẽ subplot
    nexttile; hold on;
    plot(x, y_deepaco, 'Color',[0.85 0.1 0.1],'LineWidth',1.2,'DisplayName','DeepACO');
    plot(x, y_gfacs,   'Color',[0.1 0.3 0.7],'LineWidth',1.2,'DisplayName','GFACS');
    plot(x, y_ppo,     'Color',[0.95 0.6 0.1],'LineWidth',1.2,'DisplayName','Neu-FACO');

    % Format
    grid on; box on;
    set(gca,'TickLabelInterpreter','latex','FontSize',8);
    %xlabel('Iteration','Interpreter','latex','FontSize',10);
    %ylabel('Cost','Interpreter','latex','FontSize',10);
    title(inst,'Interpreter','latex','FontSize',10);

    % Giới hạn số tick trên trục Y (6–7 giá trị)
    ax = gca;
    ax.YTick = linspace(min(ax.YLim), max(ax.YLim), 6);

    % Giới hạn số tick trên trục X (6–7 giá trị)
    ax.XTick = linspace(min(ax.XLim), max(ax.XLim), 6);

    % Legend cho tất cả subplot
    legend('Location','northeast','Interpreter','latex','FontSize',7);
end

% ==== Xuất file PNG đồng bộ size ====
set(gcf,'Units','inches','Position',[1 1 12 9]);   % Figure tổng thể
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 12 9]);
print(gcf, 'TSP_9instances.png','-dpng','-r300');
