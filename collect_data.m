%% ========================================================================
%  DATA COLLECTION SCRIPT FOR ML TRAINING
%  Generates training data for transmitter localization
%  ========================================================================

clear all;
close all;
clc;

%% ========================================================================
% Data Collection Parameters
% =========================================================================

% Distance range (µm) - reasonable distances from origin
distances = [20, 30, 40, 50, 60, 70, 80];

% Angles (degrees) - cover all quadrants evenly
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315];

% Number of trials per configuration (for robustness)
numTrials = 5;

% Simulation parameters (same as main.m)
D = 100;            % diffusion coefficient (um^2/s)
deltat = 0.01;      % time step (s)
T = 100;            % total time (s)
N = 500;            % number of molecules
r = 10;             % receiver radius (µm)

%% ========================================================================
% Generate Configuration Grid
% =========================================================================

configCount = 0;
configs = [];

for d = distances
    for a = angles_deg
        % Convert polar to Cartesian
        x0 = d * cosd(a);
        y0 = d * sind(a);
        
        configCount = configCount + 1;
        configs(configCount).x0 = x0;
        configs(configCount).y0 = y0;
        configs(configCount).distance = d;
        configs(configCount).angle = a;
    end
end

totalSims = configCount * numTrials;
fprintf('=== Data Collection Plan ===\n');
fprintf('Configurations: %d\n', configCount);
fprintf('Trials per config: %d\n', numTrials);
fprintf('Total simulations: %d\n', totalSims);
fprintf('================================\n\n');

%% ========================================================================
% Data Collection Loop
% =========================================================================

% Preallocate data storage
allData = struct('x0', {}, 'y0', {}, 'trial', {}, ...
                 'n_absorbed', {}, 'absorption_times', {}, 'impact_angles', {});

dataIdx = 1;
sigma = sqrt(2 * D * deltat);
t = 0:deltat:T;
numSteps = length(t);

for configIdx = 1:configCount
    x0 = configs(configIdx).x0;
    y0 = configs(configIdx).y0;
    
    fprintf('Config %d/%d: (x0=%.1f, y0=%.1f) | Distance=%.1f µm, Angle=%.0f°\n', ...
            configIdx, configCount, x0, y0, configs(configIdx).distance, configs(configIdx).angle);
    
    for trial = 1:numTrials
        % Set random seed for reproducibility
        rng(configIdx * 1000 + trial);
        
        % Initialize molecule positions
        X = zeros(numSteps, N);
        Y = zeros(numSteps, N);
        X(1, :) = x0;
        Y(1, :) = y0;
        
        isAbsorbed = false(1, N);
        absorptionTime = NaN(1, N);
        absorptionTimeIndex = NaN(1, N);
        
        % Run simulation
        for j = 1:N
            for i = 2:numSteps
                if isAbsorbed(j)
                    X(i, j) = X(i-1, j);
                    Y(i, j) = Y(i-1, j);
                else
                    % Brownian step
                    X(i, j) = X(i-1, j) + randn(1, 1) * sigma;
                    Y(i, j) = Y(i-1, j) + randn(1, 1) * sigma;
                    
                    % Check absorption
                    distance = sqrt(X(i, j)^2 + Y(i, j)^2);
                    if distance <= r
                        isAbsorbed(j) = true;
                        absorptionTime(j) = t(i);
                        absorptionTimeIndex(j) = i;
                    end
                end
            end
        end
        
        % Compute impact angles
        impactAngle = NaN(1, N);
        for j = 1:N
            if isAbsorbed(j)
                absIdx = absorptionTimeIndex(j);
                impactAngle(j) = atan2(Y(absIdx, j), X(absIdx, j));
            end
        end
        
        % Store data for absorbed molecules only
        absorbedMask = isAbsorbed;
        n_absorbed = sum(absorbedMask);
        
        allData(dataIdx).x0 = x0;
        allData(dataIdx).y0 = y0;
        allData(dataIdx).trial = trial;
        allData(dataIdx).n_absorbed = n_absorbed;
        allData(dataIdx).absorption_times = absorptionTime(absorbedMask);
        allData(dataIdx).impact_angles = impactAngle(absorbedMask);
        
        fprintf('  Trial %d: %d molecules absorbed\n', trial, n_absorbed);
        dataIdx = dataIdx + 1;
    end
    fprintf('\n');
end

%% ========================================================================
% Save Data
% =========================================================================

% Save as MATLAB .mat file
save('ml_training_data.mat', 'allData', 'configs', 'D', 'deltat', 'T', 'N', 'r');
fprintf('Data saved to: ml_training_data.mat\n');

% Also export to CSV format (one row per molecule)
fprintf('Exporting to CSV format...\n');
csvData = [];

for i = 1:length(allData)
    x0 = allData(i).x0;
    y0 = allData(i).y0;
    trial = allData(i).trial;
    n = allData(i).n_absorbed;
    
    for j = 1:n
        row = [x0, y0, trial, allData(i).absorption_times(j), allData(i).impact_angles(j)];
        csvData = [csvData; row];
    end
end

% Write CSV
csvHeader = 'x0,y0,trial,absorption_time,impact_angle';
fid = fopen('ml_training_data.csv', 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite('ml_training_data.csv', csvData, '-append', 'precision', '%.6f');

fprintf('Data saved to: ml_training_data.csv\n');
fprintf('Total data points (molecules): %d\n', size(csvData, 1));
fprintf('\n=== Data Collection Complete ===\n');
