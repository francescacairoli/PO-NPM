function [ compact_bin_hist, bin_cum_dist, bin_size ] = estimate_state_distribution( sim_fun, n, time, params, state_space, init_region, resolution )
%ESTIMATE_STATE_DISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here

bin_size = 1;
d = size(state_space, 1);
for i = 1:d
    bin_size = bin_size * ceil((state_space(i, 2) - state_space(i, 1)) / resolution(i)); 
end
fprintf('bin_size: %d\n', bin_size);
bin_hist = zeros(1, bin_size);

init_states = zeros(d + 1, n);
for i = 1:n
    init_states(:, i) = rand_state(init_region);
end

%states = [];
gen_clock = tic;
for i = 1:n
    if exist('params', 'var') && ~isempty(params)
        tr = sim_fun(init_states(:,i), time, params);
    else
        tr = sim_fun(init_states(:,i), time);
    end
    bin_ind = find_bin(tr, state_space, resolution);
    %bin_ind = bin_ind(bin_ind <= length(bin_hist));
    bin_hist(bin_ind) = bin_hist(bin_ind) + 1;
    %states = [states tr];
   % plot(t, tr(1,:)); hold on;
    if mod(i, 500) == 0
        gen_time = round(toc(gen_clock));
        fprintf('%i trajectories generated. Elapsed time: %d:%d:%d\n', i, floor(gen_time / 3600), floor(mod(gen_time, 3600) / 60), mod(gen_time, 60));
    end
end

% compact bin_hist
compact_bin_hist = [];
non_zero_ind = find(bin_hist > 0);
if ~isempty(non_zero_ind)
    % first row: linear indices of the bins
    % second row: bin frequencies
    compact_bin_hist = [non_zero_ind'; bin_hist(non_zero_ind)'];
end

% compact bin cumulative distribution
bin_cum_dist = [];
if ~isempty(compact_bin_hist)
    bin_cum_dist = [compact_bin_hist(1,:); cumsum(compact_bin_hist(1,:))];
    % first row: linear indices of the bins
    % second row: cumulative probabilities
    bin_cum_dist(2,:) = bin_cum_dist(2,:) / bin_cum_dist(2,end);
end

    function bin_ind = find_bin(x, range, resolution)
        fb_n = size(x, 2);
        var_ind = max(floor((x(1:7,:) - repmat(range(:, 1), 1, fb_n)) ./ repmat(resolution, 1, fb_n)), 0) + 1;
        mat_size = ceil((range(:,2) - range(:,1)) ./ resolution)';
        filter = var_ind(1,:) <= mat_size(1);
        for fb_i = 2:7
            filter = filter & var_ind(fb_i,:) <= mat_size(fb_i);
        end
        var_ind = var_ind(:,filter);
        bin_ind = sub2ind(mat_size, var_ind(1,:), var_ind(2,:), var_ind(3,:), var_ind(4,:), var_ind(5,:), var_ind(6,:), var_ind(7,:));
    end

end


