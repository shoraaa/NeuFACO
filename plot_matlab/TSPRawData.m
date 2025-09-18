% ================== Setup ==================
clear; clc;

files = { ... 
  % DeepACO
  'test_result_ckptdeepaco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptdeepaco_200-tsplib200-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsp1000-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsplib500-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptdeepaco_500-tsplib1000-nants100-niter100-nruns3-seed0.csv', ...
  
  % GFACS
  'test_result_ckptgfacs_200-tsp200-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_200-tsplib200-nants100-niter100-nruns5-seed0.csv', ...
  'test_result_ckptgfacs_500-tsp500-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_500-tsp1000-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptgfacs_500-tsplib500-nants100-niter100-nruns3-seed0.csv', ...
  'test_result_ckptgfacs_500-tsplib1000-nants100-niter100-nruns3-seed0.csv', ...
  
  % PPO-FACO
  'test_result_ckptppo_faco_200-tsp200-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_200-tsplib200-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsp500-ninstNone-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsp1000-ninst64-AS-nants100-niter100-nruns1-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsplib500-nants100-niter100-nruns10-seed0.csv', ...
  'test_result_ckptppo_faco_500-tsplib1000-nants100-niter100-nruns3-seed0.csv' ...
};

optVals = containers.Map({'tsp200','tsp500','tsp1000'}, [10.73, 16.53, 23.14]);

Summary = table('Size',[0 12], ...
  'VariableTypes',{'string','string','double','double','double','double','double','double','double','double','double','double'}, ...
  'VariableNames',{'Dataset','Method', ...
                   'Error_Min_%','Error_Q1_%','Error_Median_%','Error_Q3_%', ...
                   'Whisker_Low_%','Whisker_High_%', ...
                   'Error_Mean_%','Error_Max_%', ...
                   'Avg_Time','Avg_Cost'});

for f = 1:numel(files)
  fname = files{f};
  T = readtable(fname,'VariableNamingRule','preserve');

  % ----- Detect dataset -----
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
      method = "NeuFACO";
  end

  % ----- Lấy vector error -----
  if startsWith(dataset,"TSPLib")
      errors = T.("Error_Mean_%");
  else
      errors = abs(T.mean_cost - opt) ./ opt * 100;
  end
  errors = errors(~isnan(errors));

  q1  = prctile(errors,25);
  med = median(errors);
  q3  = prctile(errors,75);
  iqr = q3 - q1;
  lowerBound = q1 - 1.5*iqr;
  upperBound = q3 + 1.5*iqr;
  el = min(errors(errors >= lowerBound)); if isempty(el), el = min(errors); end
  eh = max(errors(errors <= upperBound)); if isempty(eh), eh = max(errors); end

  emin  = min(errors);
  emean = mean(errors);
  emax  = max(errors);

  % ----- Avg_Time -----
  if ismember("avg_time", T.Properties.VariableNames)
      avgTime = mean(T.avg_time,'omitnan');
  elseif ismember("Avg_Time", T.Properties.VariableNames)
      avgTime = mean(T.Avg_Time,'omitnan');
  else
      avgTime = NaN;
  end

  % ----- Avg_Cost -----
  if ismember("mean_cost", T.Properties.VariableNames)
      avgCost = mean(T.mean_cost,'omitnan');
  elseif ismember("Length_Mean", T.Properties.VariableNames)
      avgCost = mean(T.Length_Mean,'omitnan');
  else
      avgCost = NaN;
  end

  % ----- Append -----
  Summary = [Summary; {dataset, method, ...
                       emin, q1, med, q3, ...
                       el, eh, ...
                       emean, emax, ...
                       avgTime, avgCost}];
end

writetable(Summary,'Error_Summary_Calc.csv');
disp('File Error_Summary_Calc.csv đã được tạo.');
