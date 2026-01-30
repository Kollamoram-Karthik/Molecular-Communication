%% ========================================================================
%  3D MOLECULAR COMMUNICATION SIMULATION
%  Point Tx to Spherical Absorbing Rx
%  DATASET GENERATION FOR ML TRAINING
%  ========================================================================

%% ========================================================================
% Initialize Parameters
% =========================================================================

clear all;
close all;
clc;

D = 100;            % diffusion coefficient (um^2/s)
deltat = 0.01;      % time step (s)
T = 100;            % tot time (s)
N = 2000;           % no of molecules (INCREASED from 500)

% Rx (at origin)
r = 10;             

% Dataset generation parameters
num_samples = 5000;  % Number of different (x0, y0) configurations (INCREASED from 1000)
x0_min = 15;         % Minimum x0 (> 10 um)
x0_max = 90;         % Maximum x0
y0_min = 15;         % Minimum y0 (> 10 um)
y0_max = 90;         % Maximum y0

% Heatmap generation parameters
time_bins = 100;     % Number of time bins (high resolution)
angle_bins = 100;    % Number of angle bins (high resolution)
time_min = 0;        % Minimum time (s)
time_max = T;        % Maximum time (s)
angle_min = -pi;     % Minimum angle (radians)
angle_max = pi;      % Maximum angle (radians)

% Generate random (x0, y0) pairs
rng(42);  % For reproducibility
x0_samples = x0_min + (x0_max - x0_min) * rand(num_samples, 1);
y0_samples = y0_min + (y0_max - y0_min) * rand(num_samples, 1);

% Preallocate storage for dataset
dataset = cell(num_samples, 1);  % Each cell will contain data for one simulation

t = 0:deltat:T;
numSteps = length(t);
sigma = sqrt(2 * D * deltat);

fprintf('=== Starting Dataset Generation ===\n');
fprintf('Total simulations: %d\n', num_samples);
fprintf('x0 range: %.1f - %.1f um\n', x0_min, x0_max);
fprintf('y0 range: %.1f - %.1f um\n', y0_min, y0_max);
fprintf('Molecules per simulation: %d\n', N);
fprintf('Time window: %.1f s\n', T);
fprintf('Heatmap resolution: %d x %d (time x angle)\n\n', time_bins, angle_bins);

%% ========================================================================
% Dataset Generation Loop
% =========================================================================

for sample_idx = 1:num_samples
    % Current transmitter position
    x0 = x0_samples(sample_idx);
    y0 = y0_samples(sample_idx);
    
    % Initialize molecule positions
    X = zeros(numSteps, N);
    Y = zeros(numSteps, N);
    X(1, :) = x0;
    Y(1, :) = y0;
    
    isAbsorbed = false(1, N);           
    absorptionTime = NaN(1, N);
    absorptionTimeIndex = NaN(1, N);
    
    % Run simulation (core algorithm unchanged)
    for j = 1:N
        for i = 2:numSteps
            if isAbsorbed(j)
                % Already absorbed, stays at absorption point
                X(i, j) = X(i-1, j);
                Y(i, j) = Y(i-1, j);
            else
                % Gaussian Step
                X(i, j) = X(i-1, j) + randn(1, 1) * sigma;
                Y(i, j) = Y(i-1, j) + randn(1, 1) * sigma;
                
                % Check if absorbed or not
                distance = sqrt(X(i, j)^2 + Y(i, j)^2);
                if distance <= r
                    isAbsorbed(j) = true;
                    absorptionTime(j) = t(i);
                    absorptionTimeIndex(j) = i;
                end
            end
        end
    end
    
    % Extract absorbed molecules data
    absorbed_indices = find(isAbsorbed);
    N0 = length(absorbed_indices);
    
    % Calculate impact angles for absorbed molecules
    impact_angles = NaN(N0, 1);
    absorption_times = NaN(N0, 1);
    
    for k = 1:N0
        j = absorbed_indices(k);
        idx = absorptionTimeIndex(j);
        
        % Impact angle: angle at which molecule hits receiver sphere
        impact_angles(k) = atan2(Y(idx, j), X(idx, j));
        absorption_times(k) = absorptionTime(j);
    end
    
    % Generate heatmap for this sample
    heatmap = generate_heatmap(absorption_times, impact_angles, ...
        time_bins, angle_bins, time_min, time_max, angle_min, angle_max);
    
    % Store data for this simulation
    dataset{sample_idx} = struct(...
        'x0', x0, ...
        'y0', y0, ...
        'distance', sqrt(x0^2 + y0^2), ...
        'N0', N0, ...
        'absorption_times', absorption_times, ...
        'impact_angles', impact_angles, ...
        'heatmap', heatmap);
    
    % Progress update
    if mod(sample_idx, 100) == 0
        fprintf('Progress: %d/%d simulations complete (%.1f%%)\n', ...
            sample_idx, num_samples, 100*sample_idx/num_samples);
    end
end

% Display dataset summary
fprintf('\n=== Dataset Generation Complete ===\n');
fprintf('Total samples: %d\n', num_samples);
fprintf('Average N0: %.1f molecules\n', mean(cellfun(@(x) x.N0, dataset)));
fprintf('Min N0: %d molecules\n', min(cellfun(@(x) x.N0, dataset)));
fprintf('Max N0: %d molecules\n', max(cellfun(@(x) x.N0, dataset)));
fprintf('Heatmap shape: %d x %d\n', time_bins, angle_bins);

% Save dataset
save('molecular_comm_dataset.mat', 'dataset', 'x0_samples', 'y0_samples', ...
     'D', 'deltat', 'T', 'N', 'r', 'time_bins', 'angle_bins', ...
     'time_min', 'time_max', 'angle_min', 'angle_max', '-v7.3');
fprintf('\nDataset saved to: molecular_comm_dataset.mat\n');

%% ========================================================================
% OPTIONAL: Visualization of Sample Simulation
% =========================================================================
% Uncomment this section to visualize a single random sample from the dataset

% sample_to_visualize = randi(num_samples);
% sample_data = dataset{sample_to_visualize};
% 
% fprintf('\n=== Visualizing Sample %d ===\n', sample_to_visualize);
% fprintf('Position: (%.2f, %.2f) um\n', sample_data.x0, sample_data.y0);
% fprintf('Distance: %.2f um\n', sample_data.distance);
% fprintf('Molecules absorbed: %d\n', sample_data.N0);
% 
% if sample_data.N0 > 0
%     figure('Name', sprintf('Sample %d Visualization', sample_to_visualize), 'NumberTitle', 'off');
%     
%     % Plot impact angles on unit circle
%     subplot(1, 2, 1);
%     theta_circle = linspace(0, 2*pi, 100);
%     plot(cos(theta_circle), sin(theta_circle), 'k-', 'LineWidth', 2);
%     hold on;
%     for k = 1:sample_data.N0
%         angle = sample_data.impact_angles(k);
%         plot(cos(angle), sin(angle), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
%     end
%     axis equal;
%     grid on;
%     xlabel('X', 'FontSize', 12);
%     ylabel('Y', 'FontSize', 12);
%     title('Impact Angles on Receiver Sphere', 'FontSize', 14);
%     hold off;
%     
%     % Plot absorption time histogram
%     subplot(1, 2, 2);
%     histogram(sample_data.absorption_times, 20, 'FaceColor', 'g', 'EdgeColor', 'k');
%     xlabel('Time (s)', 'FontSize', 12);
%     ylabel('Number of Molecules', 'FontSize', 12);
%     title('Absorption Time Distribution', 'FontSize', 14);
%     grid on;
% end

%%

%% ========================================================================
% HEATMAP GENERATION FUNCTION
% =========================================================================
function heatmap = generate_heatmap(absorption_times, impact_angles, ...
    time_bins, angle_bins, time_min, time_max, angle_min, angle_max)
    % Generate 2D histogram (heatmap) of absorption times vs impact angles
    % 
    % Inputs:
    %   absorption_times: Vector of absorption times (seconds)
    %   impact_angles: Vector of impact angles (radians, -pi to pi)
    %   time_bins: Number of bins for time axis
    %   angle_bins: Number of bins for angle axis
    %   time_min, time_max: Time range for binning
    %   angle_min, angle_max: Angle range for binning
    %
    % Output:
    %   heatmap: 2D matrix of size [time_bins x angle_bins]
    %            where each element contains the count of molecules
    %            that arrived at that specific time-angle bin
    
    % Handle empty input (no molecules absorbed)
    if isempty(absorption_times) || isempty(impact_angles)
        heatmap = zeros(time_bins, angle_bins);
        return;
    end
    
    % Define bin edges
    time_edges = linspace(time_min, time_max, time_bins + 1);
    angle_edges = linspace(angle_min, angle_max, angle_bins + 1);
    
    % Generate 2D histogram using histcounts2
    % Note: histcounts2 returns counts in format [length(xedges)-1, length(yedges)-1]
    % We use absorption_times as X (rows) and impact_angles as Y (columns)
    heatmap = histcounts2(absorption_times, impact_angles, time_edges, angle_edges);
    
    % heatmap is now [time_bins x angle_bins] where:
    % - Rows represent time bins (0 to T)
    % - Columns represent angle bins (-pi to pi)
    % - Each element is the count of molecules in that bin
end
