%% ========================================================================
%  3D MOLECULAR COMMUNICATION SIMULATION
%  Point Tx to Spherical Absorbing Rx
%  ========================================================================

%% ========================================================================
% Initialize Parameters
% =========================================================================

clear all;
close all;
clc;

D = 100;            % diffusion coefficient (um^2/s)
deltat = 0.01;      % time step (s)
T = 100;             % tot time (s)
N = 500;             % no of molecules

% Tx (um)
x0 = 30;            
y0 = 30;            
z0 = 30; 

% Rx (at origin)
r = 10;             


t = 0:deltat:T;
numSteps = length(t);


X = zeros(numSteps, N);
Y = zeros(numSteps, N);
Z = zeros(numSteps, N);

X(1, :) = x0;
Y(1, :) = y0;
Z(1, :) = z0;

isAbsorbed = false(1, N);           
absorptionTime = NaN(1, N);
absorptionTimeIndex = NaN(1, N);

sigma = sqrt(2 * D * deltat);

%% ========================================================================
% Algorithm
% =========================================================================

for j = 1:N
    for i = 2:numSteps
        if isAbsorbed(j)
            % 
            X(i, j) = X(i-1, j);
            Y(i, j) = Y(i-1, j);
            Z(i, j) = Z(i-1, j);
        else
            % Guassian Step
            X(i, j) = X(i-1, j) + randn(1, 1) * sigma;
            Y(i, j) = Y(i-1, j) + randn(1, 1) * sigma;
            Z(i, j) = Z(i-1, j) + randn(1, 1) * sigma;
            
            % Check if absorbed or not
            distance = sqrt(X(i, j)^2 + Y(i, j)^2 + Z(i, j)^2);
            if distance <= r
                isAbsorbed(j) = true;
                absorptionTime(j) = t(i);
                absorptionTimeIndex(j) = i;
            end
        end
    end
end

% Display simulation summary
fprintf('=== Simulation Complete ===\n');
fprintf('Molecules Emitted: %d\n', N);
fprintf('Time Window: %d\n', T);
fprintf('Molecules Absorbed: %d\n', sum(isAbsorbed));
fprintf('Probability: %.2f%%\n', 100 * sum(isAbsorbed) / N);

%% ========================================================================
% X vs T
% =========================================================================

figure('Name', 'X Position vs Time', 'NumberTitle', 'off');
hold on;
grid on;

for j = 1:N
    if isAbsorbed(j)
        % Plot trajectory up to absorption in red, then green
        absIdx = absorptionTimeIndex(j);
        plot(t(1:absIdx), X(1:absIdx, j), 'r-', 'LineWidth', 0.5);
        plot(t(absIdx:end), X(absIdx:end, j), 'g-', 'LineWidth', 1.5);
        % Mark absorption point
        plot(t(absIdx), X(absIdx, j), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    else
        % Non-absorbed molecule stays red
        plot(t, X(:, j), 'r-', 'LineWidth', 0.5);
    end
end

xlabel('Time (s)', 'FontSize', 12);
ylabel('X Position (\mum)', 'FontSize', 12);
title('X Position vs Time for All Molecules', 'FontSize', 14);
legend({'Red: Not absorbed', 'Green: Absorbed'}, 'Location', 'best');
hold off;

%% ========================================================================
% Y vs T 
% =========================================================================

figure('Name', 'Y Position vs Time', 'NumberTitle', 'off');
hold on;
grid on;

for j = 1:N
    if isAbsorbed(j)
        absIdx = absorptionTimeIndex(j);
        plot(t(1:absIdx), Y(1:absIdx, j), 'r-', 'LineWidth', 0.5);
        plot(t(absIdx:end), Y(absIdx:end, j), 'g-', 'LineWidth', 1.5);
        plot(t(absIdx), Y(absIdx, j), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    else
        plot(t, Y(:, j), 'r-', 'LineWidth', 0.5);
    end
end

xlabel('Time (s)', 'FontSize', 12);
ylabel('Y Position (\mum)', 'FontSize', 12);
title('Y Position vs Time for All Molecules', 'FontSize', 14);
legend({'Red: Not absorbed', 'Green: Absorbed'}, 'Location', 'best');
hold off;

%% ========================================================================
% Z vs T
% =========================================================================

figure('Name', 'Z Position vs Time', 'NumberTitle', 'off');
hold on;
grid on;

for j = 1:N
    if isAbsorbed(j)
        absIdx = absorptionTimeIndex(j);
        plot(t(1:absIdx), Z(1:absIdx, j), 'r-', 'LineWidth', 0.5);
        plot(t(absIdx:end), Z(absIdx:end, j), 'g-', 'LineWidth', 1.5);
        plot(t(absIdx), Z(absIdx, j), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    else
        plot(t, Z(:, j), 'r-', 'LineWidth', 0.5);
    end
end

xlabel('Time (s)', 'FontSize', 12);
ylabel('Z Position (\mum)', 'FontSize', 12);
title('Z Position vs Time for All Molecules', 'FontSize', 14);
legend({'Red: Not absorbed', 'Green: Absorbed'}, 'Location', 'best');
hold off;

%% ========================================================================
% X vs Y
% =========================================================================

figure('Name', 'X-Y Plane Animation', 'NumberTitle', 'off');

xyLimit = max([abs(x0), abs(y0)]) * 1.3;

theta_circle = linspace(0, 2*pi, 100);
rx_circle = r * cos(theta_circle);
ry_circle = r * sin(theta_circle);

% Animation loop
for i = 1:numSteps
    clf;
    hold on;
    grid on;
    axis equal;
    xlim([x0/2 - xyLimit, x0/2 + xyLimit]);
    ylim([y0/2 - xyLimit, y0/2 + xyLimit]);
    
    % Draw receiver (blue circle)
    fill(rx_circle, ry_circle, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
    
    % Plot current position of each molecule
    for j = 1:N
        if isAbsorbed(j) && i >= absorptionTimeIndex(j)
            % Absorbed molecule - green
            plot(X(i, j), Y(i, j), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        else
            % Not yet absorbed - red
            plot(X(i, j), Y(i, j), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end
    
    % Mark transmitter location
    plot(x0, y0, 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    
    xlabel('X Position (\mum)', 'FontSize', 12);
    ylabel('Y Position (\mum)', 'FontSize', 12);
    title(sprintf('X-Y Plane | Time: %.2f s | Absorbed: %d/%d', t(i), sum(isAbsorbed & (absorptionTimeIndex <= i)), N), 'FontSize', 14);
    
    drawnow;
    hold off;
end

%% ========================================================================
% Y vs Z
% =========================================================================

figure('Name', 'Y-Z Plane Animation', 'NumberTitle', 'off');
yzLimit = max([abs(y0), abs(z0)]) * 1.3;

for i = 1:numSteps
    clf;
    hold on;
    grid on;
    axis equal;
    xlim([y0/2 - yzLimit, y0/2 + yzLimit]);
    ylim([z0/2 - yzLimit, z0/2 + yzLimit]);
    
    fill(ry_circle, r * sin(theta_circle), 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
    
    for j = 1:N
        if isAbsorbed(j) && i >= absorptionTimeIndex(j)
            plot(Y(i, j), Z(i, j), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        else
            plot(Y(i, j), Z(i, j), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end
    
    plot(y0, z0, 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    
    xlabel('Y Position (\mum)', 'FontSize', 12);
    ylabel('Z Position (\mum)', 'FontSize', 12);
    title(sprintf('Y-Z Plane | Time: %.2f s | Absorbed: %d/%d', t(i), sum(isAbsorbed & (absorptionTimeIndex <= i)), N), 'FontSize', 14);
    
    drawnow;
    hold off;
end

%% ========================================================================
% X vs Z
% =========================================================================

figure('Name', 'X-Z Plane Animation', 'NumberTitle', 'off');

xzLimit = max([abs(x0), abs(z0)]) * 1.3;

for i = 1:numSteps
    clf;
    hold on;
    grid on;
    axis equal;
    xlim([x0/2 - xzLimit, x0/2 + xzLimit]);
    ylim([z0/2 - xzLimit, z0/2 + xzLimit]);
    
    fill(rx_circle, r * sin(theta_circle), 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
    
    for j = 1:N
        if isAbsorbed(j) && i >= absorptionTimeIndex(j)
            plot(X(i, j), Z(i, j), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        else
            plot(X(i, j), Z(i, j), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end
    
    plot(x0, z0, 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    
    xlabel('X Position (\mum)', 'FontSize', 12);
    ylabel('Z Position (\mum)', 'FontSize', 12);
    title(sprintf('X-Z Plane | Time: %.2f s | Absorbed: %d/%d', t(i), sum(isAbsorbed & (absorptionTimeIndex <= i)), N), 'FontSize', 14);
    
    drawnow;
    hold off;
end

%% ========================================================================
% Receiver's POV
% =========================================================================

figure('Name', 'Receiver Response', 'NumberTitle', 'off');

cumulativeAbsorbed = zeros(1, numSteps);

for i = 1:numSteps
    cumulativeAbsorbed(i) = sum(absorptionTimeIndex <= i);
end

subplot(2, 1, 1);
plot(t, cumulativeAbsorbed, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Cumulative Molecules Absorbed', 'FontSize', 12);
title('Cumulative Molecules Received vs Time', 'FontSize', 14);
ylim([0, max(cumulativeAbsorbed) + 1]);

subplot(2, 1, 2);
if sum(isAbsorbed) > 0
    validAbsorptionTimes = absorptionTime(~isnan(absorptionTime));
    histogram(validAbsorptionTimes, 20, 'FaceColor', 'g', 'EdgeColor', 'k');
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Number of Molecules', 'FontSize', 12);
    title('Distribution of Absorption Times', 'FontSize', 14);
    grid on;
else
    text(0.5, 0.5, 'No molecules absorbed', 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'Units', 'normalized');
    title('Distribution of Absorption Times', 'FontSize', 14);
end

%%
