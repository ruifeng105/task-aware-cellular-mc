
file_name = '250ng';
first_pulse_index = 22;

data_cut = data(:, [times >=10 & times<=30]);
medians = median(data_cut, 2);

mean_data = mean(data);

dlmwrite([file_name, '_unnormalized.txt'], data, ' ');
dlmwrite([file_name, '_mean_unnormalized.txt'], mean_data, ' ');

data_norm = bsxfun(@rdivide,data,medians);
mean_data_norm = mean(data_norm);

plot(times, mean_data_norm);

dlmwrite([file_name, '.txt'], data_norm, ' ');
dlmwrite([file_name, '_mean.txt'], mean_data_norm, ' ');

data_norm_trunc = data_norm(:, first_pulse_index : end);
data_norm_mean_trunc = mean_data_norm(:, first_pulse_index : end);

dlmwrite([file_name, '_trunc.txt'], data_norm_trunc, ' ');
dlmwrite([file_name, '_mean_trunc.txt'], data_norm_mean_trunc, ' ');

times_trunc = times(first_pulse_index : end);
times_trunc = times_trunc - times_trunc(1);

plot(times_trunc, data_norm_mean_trunc);


