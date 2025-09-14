% ================== Thiết lập LaTeX ==================
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');

% ================== Load dữ liệu tổng hợp ==================
T = readtable('Error_Summary_Calc.csv','VariableNamingRule','preserve');

% ================== Danh sách dataset + phương pháp ==================
instances = ["TSPLib200","TSPLib500","TSPLib1000", ...
             "TSPRand200","TSPRand500","TSPRand1000"];
methods   = {'DeepACO','GFACS','NeuFACO'};
colors    = [0.75 0.15 0.15;   % đỏ cho DeepACO
             0.20 0.45 0.75;   % xanh cho GFACS
             0.90 0.55 0.20];  % cam cho NeuFACO
M = numel(methods);

% ================== Tham số bố cục cụm ==================
step   = 0.6;                % khoảng cách giữa các cluster (nhỏ -> sát)
offset = 0.20*step;           % đẩy 3 hộp trong 1 cluster
boxW   = min(0.8*(2*offset), 0.10);   % rộng mỗi hộp
capW   = 0.35*boxW;           % độ rộng "mũ" whisker

numInstances = numel(instances);
tickX = step*(1:numInstances);

% ================== Figure/Axes ==================
figure('Position',[100 100 1400 500],'Color','w');
ax = gca; hold(ax,'on');
ax.FontSize = 18;                             % cỡ chữ số trên trục
ylabel('Error [\%]','Interpreter','latex','FontSize',12);

ax.Color  = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
ax.XGrid  = 'on'; ax.YGrid = 'on';
ax.GridColor = [0.5 0.5 0.5];

% ================== Vẽ boxplot CHUẨN từ Summary ==================
yAllMin = inf; yAllMax = -inf;

for i = 1:numInstances
    cx    = tickX(i);
    pos_i = linspace(cx-offset, cx+offset, M);  % vị trí 3 method

    for m = 1:M
        row = T(strcmp(T.Dataset,instances(i)) & strcmp(T.Method,methods{m}),:);
        if isempty(row), continue; end

        q1  = row.("Error_Q1_%");
        med = row.("Error_Median_%");
        q3  = row.("Error_Q3_%");
        wl  = row.("Whisker_Low_%");
        wh  = row.("Whisker_High_%");
        mu  = row.("Error_Mean_%");        % dùng để vẽ marker mean
        % emax = row.("Error_Max_%");      % nếu cần kiểm soát ylim

        xc = pos_i(m);
        xL = xc - boxW/2; xR = xc + boxW/2;

        % --- Thân hộp: [Q1, Q3]
        patch([xL xR xR xL], [q1 q1 q3 q3], colors(m,:), ...
              'EdgeColor','k','FaceAlpha',1.0);

        % --- Median line (Q2)
        plot([xL xR],[med med],'k','LineWidth',1.3);

        % --- Whiskers (Tukey 1.5*IQR)
        plot([xc xc],[q3 wh], 'k--','LineWidth',1.1);   % whisker trên
        plot([xc-capW/2 xc+capW/2],[wh wh], 'k','LineWidth',1.1); % mũ trên
        plot([xc xc],[wl q1], 'k--','LineWidth',1.1);   % whisker dưới
        plot([xc-capW/2 xc+capW/2],[wl wl], 'k','LineWidth',1.1); % mũ dưới

        % --- Mean marker (có thể nằm ngoài hộp nếu mean > Q3)
        if ~isnan(mu)
            scatter(xc, mu, 24, 'k', 'filled', 'MarkerEdgeColor','k');
        end

        % cập nhật min/max để set ylim
        yAllMin = min([yAllMin, wl, q1, med, q3]);
        yAllMax = max([yAllMax, wh, q3, mu]);
    end
end

% ================== Trục X,Y & chia cụm ==================
xticks(tickX);
xticklabels(instances);

leftEdge  = tickX(1)  - (offset + boxW/2 + 0.15*step);
rightEdge = tickX(end) + (offset + boxW/2 + 0.15*step);
xlim([leftEdge, rightEdge]);
ylabel('Error [\%]','Interpreter','latex');

% y-limit an toàn (đỡ cắt whisker/mean)
yl = max(0, floor(yAllMin-1));
yu = ceil(yAllMax+1);
ylim([yl-1 yu]);

% vạch phân tách giữa TSPLib (3 cụm) và TSPRand (3 cụm)
xline( (tickX(3)+tickX(4))/2 , '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);

% ================== Legend (proxy patches theo màu) ==================
ph = gobjects(M,1);
for m = 1:M
    ph(m) = patch(nan, nan, colors(m,:), 'EdgeColor','k'); %#ok<AGROW>
end
lgd = legend(ph, methods, 'Interpreter','latex', ...
    'Location','northoutside','Orientation','horizontal');
set(lgd,'Color','w','EdgeColor','k','TextColor','k');
gd = legend(ph, methods, 'Interpreter','latex', ...
    'Location','northoutside','Orientation','horizontal');
lgd.FontSize = 18;                            % cỡ chữ legend
lgd.ItemTokenSize = [18,8];                   % ô màu trong legend to hơn

hold off;

% ================== Autosave ==================
print(gcf,'TSPOverview','-dpng','-r300');
