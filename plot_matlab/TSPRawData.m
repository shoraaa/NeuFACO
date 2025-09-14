% ================== Setup ==================
clear; clc;

% Danh sách file (giữ nguyên như bạn đang có)
files = { ...
  'test_result_ckptdeepaco_200-tsplib200-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsplib500-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsplib1000-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptgfacs_200-tsplib200-nants100-niter100-nruns5-seed0.csv', ...
  'test_result_ckptgfacs_500-tsplib500-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptgfacs_500-tsplib1000-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptppo_faco_200-tsplib200-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsplib500-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsplib1000-nants100-niter100-nruns3-seed0.csv', ...
  ... % TSPRandom
  'test_result_ckptdeepaco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsp1000-ninst32-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_200-tsp200-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_500-tsp500-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_500-tsp1000-ninst32-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsp1000-ninstNone-AS-nants100-niter100-nruns1-seed0.csv' ...
};

% Optimal values cho TSPRandom (dùng trị tuyệt đối)
optVals = containers.Map({'tsp200','tsp500','tsp1000'}, [10.72, 16.55, 23.12]);

% ================== Collect Results (Boxplot chuẩn) ==================
Summary = table('Size',[0 10], ...
  'VariableTypes',{'string','string','double','double','double','double','double','double','double','double'}, ...
  'VariableNames',{'Dataset','Method', ...
                   'Error_Min_%','Error_Q1_%','Error_Median_%','Error_Q3_%', ...
                   'Whisker_Low_%','Whisker_High_%', ...
                   'Error_Mean_%','Error_Max_%'});

for f = 1:numel(files)
  fname = files{f};
  T = readtable(fname,'VariableNamingRule','preserve');

  % ----- Detect dataset (không bắt "500" mơ hồ) -----
  if contains(fname,'tsplib','IgnoreCase',true)
      if     contains(fname,'tsplib200','IgnoreCase',true), dataset = "TSPLib200";
      elseif contains(fname,'tsplib500','IgnoreCase',true), dataset = "TSPLib500";
      elseif contains(fname,'tsplib1000','IgnoreCase',true), dataset = "TSPLib1000";
      else,  error('Không nhận diện được TSPLib size trong %s', fname);
      end
  else
      if     contains(fname,'tsp200','IgnoreCase',true),  dataset = "TSPRand200";  opt = optVals('tsp200');
      elseif contains(fname,'tsp500','IgnoreCase',true),  dataset = "TSPRand500";  opt = optVals('tsp500');
      elseif contains(fname,'tsp1000','IgnoreCase',true), dataset = "TSPRand1000"; opt = optVals('tsp1000');
      else,  error('Không nhận diện được TSPRandom size trong %s', fname);
      end
  end

  % ----- Detect method -----
  if     contains(fname,'deepaco','IgnoreCase',true)
      method = "DeepACO";
  elseif contains(fname,'gfacs','IgnoreCase',true)
      method = "GFACS";
  else
      method = "NeuFACO";  % ppo_faco -> NeuFACO
  end

  % ----- Lấy vector error -----
  if startsWith(dataset,"TSPLib")
      % Dùng Error_Mean_% per-instance có sẵn trong file TSPLib
      assert(ismember("Error_Mean_%", T.Properties.VariableNames), 'Thiếu Error_Mean_%% trong %s', fname);
      errors = T.("Error_Mean_%");
  else
      % TSPRandom: gap = |mean_cost - opt| / opt * 100
      assert(ismember("mean_cost", T.Properties.VariableNames), 'Thiếu mean_cost trong %s', fname);
      errors = abs(T.mean_cost - opt) ./ opt * 100;
  end
  errors = errors(~isnan(errors));
  if isempty(errors), continue; end

  % ----- Thống kê cho boxplot chuẩn -----
  q1  = prctile(errors,25);
  med = median(errors);
  q3  = prctile(errors,75);
  iqr = q3 - q1;

  lowerBound = q1 - 1.5*iqr;
  upperBound = q3 + 1.5*iqr;

  % whisker dưới: điểm nhỏ nhất vẫn >= lowerBound
  el = min(errors(errors >= lowerBound));
  if isempty(el), el = min(errors); end

  % whisker trên: điểm lớn nhất vẫn <= upperBound
  eh = max(errors(errors <= upperBound));
  if isempty(eh), eh = max(errors); end

  emin  = min(errors);
  emean = mean(errors);
  emax  = max(errors);

  % ----- Append -----
  Summary = [Summary; {dataset, method, ...
                       emin, q1, med, q3, ...
                       el, eh, ...
                       emean, emax}];
end

% ================== Save file ==================
writetable(Summary,'Error_Summary_Calc.csv');
disp('File Error_Summary_Calc.csv đã được tạo.');
