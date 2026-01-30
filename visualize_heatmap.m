%% ========================================================================
%  HEATMAP VISUALIZATION SCRIPT
%  Visualize generated time-angle heatmaps from dataset
%  ========================================================================

clear all;
close all;
clc;

% Load dataset
fprintf('Loading dataset...\n');
load('molecular_comm_dataset.mat');

% Select a few samples to visualize
num_visualize = 4;  % Visualize 4 different samples
sample_indices = randi(length(dataset), num_visualize, 1);

% Create figure with subplots
figure('Position', [100, 100, 1200, 800], 'Name', 'Heatmap Visualization');

for idx = 1:num_visualize
    sample_idx = sample_indices(idx);
    sample_data = dataset{sample_idx};
    
    subplot(2, num_visualize/2, idx);
    
    % Display heatmap
    imagesc([angle_min, angle_max], [time_min, time_max], sample_data.heatmap);
    
    % Formatting
    colorbar;
    colormap('hot');  % Use 'hot' colormap (black -> red -> yellow -> white)
    axis xy;  % Flip y-axis so time increases upward
    xlabel('Impact Angle (rad)', 'FontSize', 10);
    ylabel('Absorption Time (s)', 'FontSize', 10);
    title(sprintf('Sample %d: (%.1f, %.1f) μm, N0=%d', ...
        sample_idx, sample_data.x0, sample_data.y0, sample_data.N0), ...
        'FontSize', 11);
    
    % Add grid
    grid on;
    set(gca, 'Layer', 'top');
end

% Add overall title
sgtitle(sprintf('Time-Angle Heatmaps (%d × %d bins)', time_bins, angle_bins), ...
    'FontSize', 14, 'FontWeight', 'bold');

fprintf('\nVisualization complete!\n');
fprintf('Heatmap shape: %d × %d\n', time_bins, angle_bins);
fprintf('Memory per heatmap: %.2f KB\n', time_bins * angle_bins * 8 / 1024);
fprintf('Total memory for %d samples: %.2f MB\n', ...
    length(dataset), length(dataset) * time_bins * angle_bins * 8 / 1024^2);
