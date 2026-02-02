%% ========================================================================
%  QUICK TEST: Heatmap Generation with Small Dataset
%  Tests the heatmap generation with just 10 samples for quick verification
%  ========================================================================

clear all;
close all;
clc;

fprintf('=== QUICK HEATMAP GENERATION TEST ===\n\n');

% Simulation parameters (same as main.m)
D = 100;
deltat = 0.01;
T = 100;
N = 2000;
r = 10;

% Small test dataset
num_samples = 10;
x0_min = 15;
x0_max = 90;
y0_min = 15;
y0_max = 90;

% Heatmap parameters (high resolution)
time_bins = 100;
angle_bins = 100;
time_min = 0;
time_max = T;
angle_min = -pi;
angle_max = pi;

% Generate random positions
rng(42);
x0_samples = x0_min + (x0_max - x0_min) * rand(num_samples, 1);
y0_samples = y0_min + (y0_max - y0_min) * rand(num_samples, 1);

dataset = cell(num_samples, 1);
t = 0:deltat:T;
numSteps = length(t);
sigma = sqrt(2 * D * deltat);

fprintf('Testing with %d samples...\n', num_samples);
fprintf('Heatmap resolution: %d x %d\n\n', time_bins, angle_bins);

tic;
for sample_idx = 1:num_samples
    x0 = x0_samples(sample_idx);
    y0 = y0_samples(sample_idx);
    
    % Initialize
    X = zeros(numSteps, N);
    Y = zeros(numSteps, N);
    X(1, :) = x0;
    Y(1, :) = y0;
    
    isAbsorbed = false(1, N);
    absorptionTime = NaN(1, N);
    absorptionTimeIndex = NaN(1, N);
    
    % Simulation
    for j = 1:N
        for i = 2:numSteps
            if isAbsorbed(j)
                X(i, j) = X(i-1, j);
                Y(i, j) = Y(i-1, j);
            else
                X(i, j) = X(i-1, j) + randn(1, 1) * sigma;
                Y(i, j) = Y(i-1, j) + randn(1, 1) * sigma;
                
                distance = sqrt(X(i, j)^2 + Y(i, j)^2);
                if distance <= r
                    isAbsorbed(j) = true;
                    absorptionTime(j) = t(i);
                    absorptionTimeIndex(j) = i;
                end
            end
        end
    end
    
    % Extract data
    absorbed_indices = find(isAbsorbed);
    N0 = length(absorbed_indices);
    
    impact_angles = NaN(N0, 1);
    absorption_times = NaN(N0, 1);
    
    for k = 1:N0
        j = absorbed_indices(k);
        idx = absorptionTimeIndex(j);
        impact_angles(k) = atan2(Y(idx, j), X(idx, j));
        absorption_times(k) = absorptionTime(j);
    end
    
    % Generate heatmap
    heatmap = generate_heatmap(absorption_times, impact_angles, ...
        time_bins, angle_bins, time_min, time_max, angle_min, angle_max);
    
    % Store
    dataset{sample_idx} = struct(...
        'x0', x0, 'y0', y0, ...
        'distance', sqrt(x0^2 + y0^2), ...
        'N0', N0, ...
        'absorption_times', absorption_times, ...
        'impact_angles', impact_angles, ...
        'heatmap', heatmap);
    
    fprintf('Sample %d: (%.1f, %.1f) μm, N0=%d, heatmap non-zero=%d\n', ...
        sample_idx, x0, y0, N0, sum(heatmap(:) > 0));
end

elapsed = toc;
fprintf('\n=== Test Complete ===\n');
fprintf('Total time: %.2f seconds\n', elapsed);
fprintf('Time per sample: %.2f seconds\n', elapsed / num_samples);

% Visualize one sample
sample = dataset{1};
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
scatter(sample.absorption_times, sample.impact_angles, 20, 'filled');
xlabel('Absorption Time (s)');
ylabel('Impact Angle (rad)');
title(sprintf('Raw Data: %d molecules', sample.N0));
grid on;

subplot(1, 3, 2);
imagesc([angle_min, angle_max], [time_min, time_max], sample.heatmap);
colorbar;
colormap('hot');
axis xy;
xlabel('Impact Angle (rad)');
ylabel('Absorption Time (s)');
title('Generated Heatmap');
grid on;

subplot(1, 3, 3);
imagesc([angle_min, angle_max], [time_min, time_max], log10(sample.heatmap + 1));
colorbar;
colormap('hot');
axis xy;
xlabel('Impact Angle (rad)');
ylabel('Absorption Time (s)');
title('Log-scale Heatmap');
grid on;

sgtitle(sprintf('Sample 1: (%.1f, %.1f) μm', sample.x0, sample.y0), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Display heatmap statistics
fprintf('\n=== Heatmap Statistics ===\n');
fprintf('Shape: %d x %d\n', size(sample.heatmap, 1), size(sample.heatmap, 2));
fprintf('Min: %d, Max: %d\n', min(sample.heatmap(:)), max(sample.heatmap(:)));
fprintf('Mean: %.2f, Std: %.2f\n', mean(sample.heatmap(:)), std(sample.heatmap(:)));
fprintf('Non-zero pixels: %d (%.1f%%)\n', sum(sample.heatmap(:) > 0), ...
    100 * sum(sample.heatmap(:) > 0) / numel(sample.heatmap));
fprintf('Memory per heatmap: %.2f KB\n', numel(sample.heatmap) * 8 / 1024);

fprintf('\n✓ Test successful! Ready to run full dataset generation.\n');

%% Function definition
function heatmap = generate_heatmap(absorption_times, impact_angles, ...
    time_bins, angle_bins, time_min, time_max, angle_min, angle_max)
    
    if isempty(absorption_times) || isempty(impact_angles)
        heatmap = zeros(time_bins, angle_bins);
        return;
    end
    
    time_edges = linspace(time_min, time_max, time_bins + 1);
    angle_edges = linspace(angle_min, angle_max, angle_bins + 1);
    heatmap = histcounts2(absorption_times, impact_angles, time_edges, angle_edges);
end
