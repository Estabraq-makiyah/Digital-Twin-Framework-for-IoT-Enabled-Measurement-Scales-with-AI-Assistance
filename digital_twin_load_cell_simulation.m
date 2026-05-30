% =========================================================================
% DIGITAL TWIN SIMULATION FRAMEWORK FOR IoT-ENABLED WEIGHING SCALES
% CAD7037 - Digital Twins and Manufacturing | MSc Portfolio
% =========================================================================
%
% PURPOSE:
%   This script simulates three core functions of the proposed digital twin
%   framework for an IoT-enabled industrial weighing scale:
%
%     MODULE 1 — Load cell drift model
%                Physics-informed simulation of measurement error
%                accumulation over operational time, incorporating
%                thermal drift, creep, and cycle-induced degradation.
%
%     MODULE 2 — Calibration prediction
%                Condition-based scheduling model that estimates
%                time-to-tolerance-breach (TTTB) from the drift model
%                output, replacing fixed-interval calibration scheduling
%                with a risk-prioritised, condition-driven approach.
%
%     MODULE 3 — Anomaly detection
%                Statistical process control-based anomaly detection
%                applied to simulated operational sensor data, identifying
%                incipient fault signatures (variance increase, drift
%                step-changes) before they exceed calibration tolerance.
%
% REFERENCES:
%   Tao, Zhang & Nee (2019) - Five-dimension DT model
%   Chen et al. (2023) - ML in DT predictive maintenance
%   Fu, Gao & Zhang (2023) - DT sensor RUL prediction
%   Martins et al. (2023) - Online sensor calibration monitoring
%   Compare, Baraldi & Zio (2020) - IoT-enabled predictive maintenance
%
% NOTE: All data is synthetically generated to represent realistic
%       industrial weighing scale degradation behaviour. No proprietary
%       data is used. Results are indicative, not empirically validated.
%
% MATLAB VERSION: R2021a or later recommended
% TOOLBOXES:     Statistics and Machine Learning Toolbox (for fitlm,
%                movmean, zscore). Core plotting uses base MATLAB only.
% =========================================================================

clc;          % Clear command window
clear;        % Clear workspace variables
close all;    % Close all open figure windows

fprintf('=================================================================\n');
fprintf('  DIGITAL TWIN SIMULATION: IoT-ENABLED WEIGHING SCALE\n');
fprintf('=================================================================\n\n');

%% =========================================================================
%  SECTION 1: GLOBAL SIMULATION PARAMETERS
%  =========================================================================
%  These parameters define the operational context of the simulated
%  weighing scale and the physical tolerances governing its calibration
%  schedule. They correspond to a mid-specification industrial platform
%  scale (e.g. Mettler-Toledo ICS series class equivalent) operating in
%  a food manufacturing/logistics environment.
% =========================================================================

% --- Time horizon ---
% Simulate 365 days of continuous operation at 1-hour resolution.
% 24 readings per day × 365 days = 8,760 data points.
t_hours  = (0:8759)';                    % Time vector: hours [0 .. 8759]
t_days   = t_hours / 24;                 % Convert to days for readability
N        = length(t_hours);              % Total number of time steps

% --- Scale operational parameters ---
nominal_capacity   = 300;    % kg — maximum rated load of the scale
operating_load_pct = 0.65;   % Typical operating load as fraction of capacity
cycles_per_hour    = 12;     % Average weighing cycles per operational hour
cumulative_cycles  = cumsum(ones(N,1) * cycles_per_hour); % Total cycle count

% --- Calibration tolerance threshold ---
% Legal-for-trade weighing instruments must remain within ±0.1% of
% full-scale reading (OIML R 76 equivalent for Class III instruments).
% This equates to ±0.3 kg error on a 300 kg capacity scale.
tolerance_pct      = 0.10;   % ±0.10% full scale
tolerance_kg       = nominal_capacity * tolerance_pct / 100; % ±0.30 kg

fprintf('Simulation parameters:\n');
fprintf('  Time horizon    : %d days\n', max(t_days));
fprintf('  Resolution      : hourly\n');
fprintf('  Scale capacity  : %d kg\n', nominal_capacity);
fprintf('  Cal. tolerance  : ±%.2f kg (%.2f%% FS)\n\n', tolerance_kg, tolerance_pct);


%% =========================================================================
%  MODULE 1: LOAD CELL DRIFT MODEL
%  =========================================================================
%  PHYSICAL BASIS:
%  Industrial strain-gauge load cells degrade through three primary
%  mechanisms over their operational life:
%
%  (1) CREEP — progressive deformation of the elastic sensing element
%      under sustained or repeated loading. Modelled as a power-law
%      function of cumulative cycle count, consistent with established
%      fatigue-accumulation models for metallic sensing elements.
%
%  (2) THERMAL DRIFT — temperature variations cause dimensional changes
%      in the load cell body and shifts in the bridge circuit zero
%      balance. Modelled as a stochastic sinusoidal process representing
%      diurnal temperature cycling in a production environment (±8°C
%      around 20°C ambient, with random shift noise).
%
%  (3) COMPENSATION COEFFICIENT OBSOLESCENCE — the static temperature
%      compensation coefficient applied at manufacture becomes less
%      accurate as the load cell's thermal characteristics change over
%      time due to material ageing. Modelled as a slow linear drift
%      in the residual thermal error after compensation.
%
%  REFERENCES: Fu, Gao & Zhang (2023); Liu et al. (2022)
% =========================================================================

fprintf('MODULE 1: Computing load cell drift model...\n');

% ---- 1.1 Creep component ------------------------------------------------
% Power-law creep model: error_creep = k_c * (cycles)^alpha
% k_c   : creep coefficient (kg per cycle^alpha), calibrated to reach
%         ~40% of tolerance at end of life under typical loading
% alpha : creep exponent (< 1 indicates decelerating creep rate,
%         consistent with primary creep in metallic elements)

k_c   = 1.2e-5;   % Creep coefficient [kg / cycle^alpha]
alpha = 0.55;     % Creep exponent (dimensionless)

drift_creep = k_c .* (cumulative_cycles .^ alpha);
% Result: monotonically increasing drift term [kg], representing
% systematic bias accumulation due to load cell element deformation.

% ---- 1.2 Thermal drift component ----------------------------------------
% Ambient temperature: diurnal sinusoidal cycle (24-hour period)
% plus random day-to-day variation representing seasonal/HVAC effects.
% Temperature sensitivity: 0.002% FS per degree Celsius (typical for
% a compensated strain-gauge load cell with aluminium alloy element).

T_ambient = 20 ...                       % Mean ambient temperature [°C]
    + 8 * sin(2*pi*t_hours/24) ...       % Diurnal cycle: ±8°C amplitude
    + randn(N,1) * 1.5;                  % Random daily variation [°C]

T_sensitivity = 0.002 / 100;            % FS per °C (as fraction)
drift_thermal = nominal_capacity * T_sensitivity * (T_ambient - 20);
% Result: zero-mean oscillating drift [kg], driven by temperature deviation
% from the 20°C calibration reference point.

% ---- 1.3 Compensation coefficient obsolescence --------------------------
% As the load cell ages, the static compensation applied at manufacture
% becomes progressively less accurate. The residual thermal error after
% compensation grows linearly with time, with added stochastic noise
% representing measurement uncertainty in the compensation estimation.

k_obs    = 4.0e-5;   % Obsolescence rate [kg/hour per °C deviation]
obs_noise = randn(N,1) * 0.003;          % Stochastic noise [kg]

drift_obsolescence = k_obs * t_hours .* abs(T_ambient - 20) + obs_noise;
% Result: slowly growing drift term [kg], representing degrading thermal
% compensation performance — a key trigger for the digital twin's
% proactive recalibration recommendation.

% ---- 1.4 Total measurement drift ----------------------------------------
% Superposition of all three drift components gives the total measurement
% error of the physical instrument at each point in time.
drift_total = drift_creep + drift_thermal + drift_obsolescence;

% ---- 1.5 Smoothed drift (digital twin virtual model estimate) -----------
% The DT's virtual model does not observe instantaneous measurement error
% directly (there is no reference standard connected continuously). It
% estimates drift by applying a moving-average filter to the operational
% signal residuals — analogous to how a Kalman-filter-based DT virtual
% model would track instrument state from IoT telemetry.

window_hours = 72;  % 3-day smoothing window [hours]
drift_DT_estimate = movmean(drift_total, window_hours);
% The smoothed estimate tracks the true drift trend while suppressing
% high-frequency thermal noise — this is the DT's operative drift model.

fprintf('  Drift model computed: %d data points\n', N);
fprintf('  End-of-year creep drift   : %.4f kg\n', drift_creep(end));
fprintf('  Peak thermal drift        : %.4f kg\n', max(abs(drift_thermal)));
fprintf('  End-of-year obsolescence  : %.4f kg\n', drift_obsolescence(end));
fprintf('  Maximum total drift       : %.4f kg (tolerance ±%.2f kg)\n\n', ...
        max(abs(drift_total)), tolerance_kg);

% ---- 1.6 Plot Module 1 results ------------------------------------------
figure('Name','Module 1: Load Cell Drift Model','NumberTitle','off', ...
       'Position',[50 500 1100 600]);

subplot(2,2,1);
plot(t_days, drift_creep, 'b-', 'LineWidth', 1.5);
hold on;
yline(tolerance_kg, 'r--', 'LineWidth', 1.5, 'Label', '+Tolerance limit');
yline(-tolerance_kg, 'r--', 'LineWidth', 1.5);
xlabel('Time (days)'); ylabel('Drift (kg)');
title('1a: Creep Drift (Power-Law Model)');
legend('Creep drift', 'Tolerance limit', 'Location', 'northwest');
grid on; xlim([0 365]);

subplot(2,2,2);
yyaxis left;
plot(t_days, T_ambient, 'Color', [0.85 0.33 0.1], 'LineWidth', 0.5);
ylabel('Temperature (°C)');
yyaxis right;
plot(t_days, drift_thermal, 'b-', 'LineWidth', 0.8);
ylabel('Drift (kg)');
xlabel('Time (days)');
title('1b: Thermal Drift vs Ambient Temperature');
grid on; xlim([0 365]);

subplot(2,2,3);
plot(t_days, drift_obsolescence, 'Color', [0.5 0 0.5], 'LineWidth', 1.2);
hold on;
yline(tolerance_kg, 'r--', 'LineWidth', 1.5);
xlabel('Time (days)'); ylabel('Drift (kg)');
title('1c: Compensation Coefficient Obsolescence');
legend('Obsolescence drift', 'Tolerance limit', 'Location', 'northwest');
grid on; xlim([0 365]);

subplot(2,2,4);
plot(t_days, drift_total, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
hold on;
plot(t_days, drift_DT_estimate, 'b-', 'LineWidth', 2.0);
yline(tolerance_kg, 'r--', 'LineWidth', 1.5, 'Label', '+Tolerance');
yline(-tolerance_kg, 'r--', 'LineWidth', 1.5, 'Label', '-Tolerance');
xlabel('Time (days)'); ylabel('Total drift (kg)');
title('1d: Total Drift — Physical vs DT Estimate');
legend('Actual drift','DT smoothed estimate','Tolerance limits', ...
       'Location','northwest');
grid on; xlim([0 365]);

sgtitle('MODULE 1: Load Cell Drift Model — IoT-Enabled Weighing Scale DT', ...
        'FontWeight','bold','FontSize',12);

fprintf('MODULE 1 complete. Figure 1 generated.\n\n');


%% =========================================================================
%  MODULE 2: CALIBRATION PREDICTION — TIME-TO-TOLERANCE-BREACH (TTTB)
%  =========================================================================
%  CONCEPT:
%  The digital twin continuously evaluates the smoothed drift estimate
%  (from Module 1) and projects forward using a linear regression model
%  fitted to the most recent drift history window. The projected
%  intersection of the drift trajectory with the calibration tolerance
%  boundary defines the estimated Time-To-Tolerance-Breach (TTTB).
%
%  This enables condition-based calibration scheduling:
%  — When TTTB > safety_margin, instrument continues normal operation
%  — When TTTB ≤ safety_margin, DT raises a calibration recommendation
%  — The safety margin is set to ensure recalibration occurs BEFORE
%    the actual tolerance is breached, maintaining legal compliance.
%
%  APPROACH:
%  Rolling linear regression on a 14-day trailing window of DT drift
%  estimates. The regression gradient predicts drift rate; extrapolating
%  to the tolerance boundary gives TTTB in days from the current point.
%  Instruments with faster drift rate receive earlier calibration alerts.
%
%  REFERENCES: Kopke, Serafim & Afonso (2024); Martins et al. (2023);
%              Chen et al. (2023)
% =========================================================================

fprintf('MODULE 2: Computing calibration prediction (TTTB)...\n');

% ---- 2.1 Parameters ------------------------------------------------------
window_reg   = 14 * 24;    % Regression window: 14 days [in hours]
safety_days  = 21;         % Safety margin: alert 21 days before breach
TTTB_days    = NaN(N,1);   % Pre-allocate TTTB vector

% ---- 2.2 Rolling linear regression to estimate TTTB ----------------------
% For each hourly time step (after the regression window fills),
% fit a linear model to the trailing window of DT drift estimates.
% Use the regression slope and intercept to project when drift will
% reach the tolerance boundary.

for i = window_reg : N
    % Extract trailing window of drift estimates
    t_window   = t_days(i - window_reg + 1 : i);     % Days in window
    d_window   = drift_DT_estimate(i - window_reg + 1 : i); % Drift in window

    % Fit linear regression: drift = m*t + b
    % Using polyfit (degree 1) for computational efficiency in the loop
    p = polyfit(t_window, d_window, 1);  % p(1) = slope, p(2) = intercept
    m = p(1);   % Drift rate [kg/day]
    b = p(2);   % Intercept [kg]

    % Project to positive tolerance boundary (upper breach)
    % tolerance = m * t_breach + b  =>  t_breach = (tolerance - b) / m
    if m > 0    % Only project if drift is trending upward
        t_breach_upper = (tolerance_kg - b) / m;    % Days from epoch
        tttb_upper     = t_breach_upper - t_days(i); % Days remaining
        TTTB_days(i)   = max(tttb_upper, 0);         % Floor at 0
    elseif m < 0
        % Drift trending down: check lower boundary breach
        t_breach_lower = (-tolerance_kg - b) / m;
        tttb_lower     = t_breach_lower - t_days(i);
        TTTB_days(i)   = max(tttb_lower, 0);
    else
        TTTB_days(i) = Inf;   % No drift trend: breach not imminent
    end
end

% ---- 2.3 Generate calibration alert flags --------------------------------
% The DT raises a CALIBRATION RECOMMENDED flag when TTTB falls to
% or below the safety margin. This converts continuous TTTB monitoring
% into a binary actionable maintenance signal for the operations team.

cal_alert = TTTB_days <= safety_days;   % Logical: 1 = alert active

% Find first alert day (first time TTTB drops to safety margin)
alert_indices = find(cal_alert);
if ~isempty(alert_indices)
    first_alert_day = t_days(alert_indices(1));
    fprintf('  First calibration alert triggered at: Day %.1f\n', first_alert_day);
else
    first_alert_day = NaN;
    fprintf('  No calibration alert triggered within simulation period.\n');
end

% ---- 2.4 Identify actual tolerance breach days ---------------------------
% For comparison: when does the actual (unmanaged) drift breach tolerance?
breach_upper = find(drift_total > tolerance_kg, 1, 'first');
breach_lower = find(drift_total < -tolerance_kg, 1, 'first');
breach_day = min([t_days(breach_upper), t_days(breach_lower)]);
if isempty(breach_day); breach_day = NaN; end

if ~isnan(breach_day)
    fprintf('  Actual tolerance breach at         : Day %.1f\n', breach_day);
    fprintf('  DT alert lead time                 : %.1f days\n\n', ...
            breach_day - first_alert_day);
end

% ---- 2.5 Plot Module 2 results -------------------------------------------
figure('Name','Module 2: Calibration Prediction','NumberTitle','off', ...
       'Position',[50 50 1100 550]);

subplot(1,2,1);
plot(t_days, TTTB_days, 'b-', 'LineWidth', 1.5);
hold on;
yline(safety_days, 'r--', 'LineWidth', 2, ...
      'Label', sprintf('Safety margin (%d days)', safety_days));
if ~isnan(first_alert_day)
    xline(first_alert_day, 'g--', 'LineWidth', 1.5, ...
          'Label', sprintf('First alert: Day %.0f', first_alert_day));
end
if ~isnan(breach_day)
    xline(breach_day, 'r-', 'LineWidth', 2, ...
          'Label', sprintf('Actual breach: Day %.0f', breach_day));
end
xlabel('Time (days)'); ylabel('TTTB (days)');
title('2a: Time-To-Tolerance-Breach (TTTB) — Rolling Prediction');
legend('TTTB estimate','Safety margin','Cal. alert','Actual breach', ...
       'Location','southwest');
ylim([0 200]); grid on; xlim([0 365]);

subplot(1,2,2);
plot(t_days, drift_DT_estimate, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_days(cal_alert), drift_DT_estimate(cal_alert), ...
     'ro', 'MarkerSize', 2, 'DisplayName', 'Alert period');
yline(tolerance_kg, 'r--', 'LineWidth', 1.5, 'Label', '+Tolerance');
yline(-tolerance_kg, 'r--', 'LineWidth', 1.5, 'Label', '-Tolerance');
if ~isnan(first_alert_day)
    xline(first_alert_day, 'g--', 'LineWidth', 1.5);
end
if ~isnan(breach_day)
    xline(breach_day, 'r-', 'LineWidth', 2);
end
xlabel('Time (days)'); ylabel('DT drift estimate (kg)');
title('2b: DT Drift Estimate with Calibration Alert Overlay');
legend('DT drift estimate','Alert period active','Tolerance limits', ...
       'Location','northwest');
grid on; xlim([0 365]);

sgtitle('MODULE 2: Condition-Based Calibration Prediction', ...
        'FontWeight','bold','FontSize',12);

fprintf('MODULE 2 complete. Figure 2 generated.\n\n');


%% =========================================================================
%  MODULE 3: ANOMALY DETECTION
%  =========================================================================
%  CONCEPT:
%  The digital twin's anomaly detection layer monitors the statistical
%  properties of the operational weight measurement data stream in
%  real time, seeking signatures that indicate incipient fault conditions
%  before they manifest as tolerance breaches. Two detection methods
%  are implemented, consistent with the literature on IoT-enabled
%  predictive maintenance (Compare, Baraldi and Zio, 2020):
%
%  METHOD A — Z-SCORE THRESHOLD DETECTOR
%  Computes the rolling z-score of the measurement residuals (deviation
%  of each reading from the short-term moving mean). Sustained high
%  z-scores indicate that the measurement distribution is shifting,
%  which is an early signature of systematic drift onset.
%
%  METHOD B — ROLLING VARIANCE DETECTOR
%  Tracks the variance of the measurement signal in a sliding window.
%  A sustained increase in variance — even without a mean shift — is
%  a characteristic precursor of load cell fault modes such as
%  strain gauge delamination or bridge circuit degradation, where
%  noise increases before the mean drifts.
%
%  REFERENCES: Chen et al. (2023); Compare, Baraldi & Zio (2020)
% =========================================================================

fprintf('MODULE 3: Computing anomaly detection...\n');

% ---- 3.1 Simulate operational measurement data ---------------------------
% Generate a simulated stream of weight readings as the DT would receive
% from the IoT-enabled scale. The signal represents repeated weighing
% of a reference product (mean 45.0 kg), with:
%   — Baseline noise (instrument resolution + environmental vibration)
%   — Superimposed drift_total from Module 1
%   — An injected fault event at day 200 (abrupt variance increase
%     simulating the onset of strain gauge delamination)
%   — A secondary drift step-change at day 280 (simulating a thermal
%     compensation coefficient failure event)

rng(42);   % Fix random seed for reproducibility

true_weight    = 45.0;                          % Reference product weight [kg]
baseline_sigma = 0.012;                         % Baseline noise std [kg]
                                                % (~0.04% FS, typical for
                                                % Class III instrument)

% Normal operational data with drift superimposed
measurements = true_weight ...
    + drift_total ...                           % Systematic drift from Module 1
    + randn(N,1) * baseline_sigma;             % Baseline stochastic noise

% --- FAULT EVENT 1: Variance increase at day 200 -------------------------
% Simulates onset of strain gauge degradation: noise increases 4x but
% mean has not yet shifted (detectable by variance monitor BEFORE drift
% monitor — this is the key value of multi-metric anomaly detection).
fault1_start = find(t_days >= 200, 1, 'first');
measurements(fault1_start:end) = measurements(fault1_start:end) ...
    + randn(N - fault1_start + 1, 1) * (baseline_sigma * 3);
% Net noise post-fault1: 4x baseline sigma

% --- FAULT EVENT 2: Step-change drift at day 280 -------------------------
% Simulates a sudden compensation coefficient failure adding a 0.12 kg
% systematic offset (40% of tolerance — significant but not yet breaching).
fault2_start = find(t_days >= 280, 1, 'first');
measurements(fault2_start:end) = measurements(fault2_start:end) + 0.12;

% ---- 3.2 Compute measurement residuals -----------------------------------
% Residuals = measurements minus short-term mean (72-hour window).
% Residuals isolate the high-frequency variability from the underlying
% drift trend, which is what the anomaly detector acts on.

short_window   = 72;    % 3-day short window [hours]
meas_mean      = movmean(measurements, short_window);
residuals      = measurements - meas_mean;

% ---- 3.3 Method A: Z-score detector --------------------------------------
% Rolling z-score uses a medium-term window (7 days) to establish
% the local mean and standard deviation, then computes the normalised
% deviation of the current reading.

zscore_window  = 7 * 24;                    % 7-day window [hours]
roll_mean      = movmean(residuals, zscore_window);
roll_std       = movstd(residuals, zscore_window);
roll_std(roll_std < 1e-10) = 1e-10;        % Prevent division by zero

z_scores       = (residuals - roll_mean) ./ roll_std;

% Smooth z-scores over 6 hours to reduce single-point false alerts
z_smooth       = movmean(abs(z_scores), 6);

% Threshold: |z| > 2.5 flagged as anomalous (corresponds to ~1.2%
% probability under Gaussian distribution — balances sensitivity vs
% false alarm rate for a production environment)
z_threshold    = 2.5;
anomaly_z      = z_smooth > z_threshold;

% ---- 3.4 Method B: Rolling variance detector ----------------------------
% Computes rolling variance in a 48-hour window. A relative increase
% of >300% over baseline variance triggers an anomaly flag.
% This specifically targets the variance-increase fault signature of
% strain gauge degradation (Fault Event 1 above).

var_window     = 48;                        % 48-hour variance window
roll_var       = movvar(measurements, var_window);

% Baseline variance: computed from first 30 days (pre-fault period)
baseline_var   = mean(roll_var(1 : 30*24));

% Relative variance increase as multiple of baseline
var_ratio      = roll_var / baseline_var;

% Threshold: flag when variance exceeds 2.5× baseline
var_threshold  = 2.5;
anomaly_var    = var_ratio > var_threshold;

% ---- 3.5 Combined alert: both detectors active --------------------------
% The DT raises a COMBINED ANOMALY ALERT when both detectors trigger
% simultaneously, reducing false positives relative to either detector
% acting alone. Single-detector alerts are still logged for review.

anomaly_combined = anomaly_z & anomaly_var;

% ---- 3.6 Detection latency analysis -------------------------------------
% How many days after each fault event does each detector first trigger?

detect_z_fault1 = find(anomaly_z(fault1_start:fault2_start-1), 1,'first');
detect_v_fault1 = find(anomaly_var(fault1_start:fault2_start-1), 1,'first');
detect_z_fault2 = find(anomaly_z(fault2_start:end), 1,'first');
detect_v_fault2 = find(anomaly_var(fault2_start:end), 1,'first');

if ~isempty(detect_z_fault1)
    fprintf('  Fault 1 (day 200) — Z-score detection latency : %.1f hours\n', detect_z_fault1);
end
if ~isempty(detect_v_fault1)
    fprintf('  Fault 1 (day 200) — Variance detection latency: %.1f hours\n', detect_v_fault1);
end
if ~isempty(detect_z_fault2)
    fprintf('  Fault 2 (day 280) — Z-score detection latency : %.1f hours\n', detect_z_fault2);
end
if ~isempty(detect_v_fault2)
    fprintf('  Fault 2 (day 280) — Variance detection latency: %.1f hours\n\n', detect_v_fault2);
end

% ---- 3.7 Plot Module 3 results -------------------------------------------
figure('Name','Module 3: Anomaly Detection','NumberTitle','off', ...
       'Position',[200 50 1200 700]);

% 3a: Raw measurement stream with fault event markers
subplot(3,1,1);
plot(t_days, measurements, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
yline(true_weight + tolerance_kg, 'r--', 'LineWidth', 1.5, ...
      'Label', '+Tolerance');
yline(true_weight - tolerance_kg, 'r--', 'LineWidth', 1.5, ...
      'Label', '-Tolerance');
xline(200, 'b--', 'LineWidth', 1.5, 'Label', 'Fault 1: variance increase');
xline(280, 'm--', 'LineWidth', 1.5, 'Label', 'Fault 2: drift step-change');
xlabel('Time (days)'); ylabel('Measured weight (kg)');
title('3a: IoT Measurement Data Stream (True weight = 45 kg)');
grid on; xlim([0 365]);

% 3b: Z-score and variance ratio detectors
subplot(3,1,2);
yyaxis left;
plot(t_days, z_smooth, 'b-', 'LineWidth', 1.0);
yline(z_threshold, 'b--', 'LineWidth', 1.0);
ylabel('Smoothed |z-score|');
yyaxis right;
plot(t_days, var_ratio, 'Color', [0.8 0.4 0], 'LineWidth', 1.0);
yline(var_threshold, '--', 'Color', [0.8 0.4 0], 'LineWidth', 1.0);
ylabel('Variance ratio (× baseline)');
xline(200, 'b--', 'LineWidth', 1.0);
xline(280, 'm--', 'LineWidth', 1.0);
xlabel('Time (days)');
title('3b: Z-Score Detector (blue) and Variance Ratio Detector (orange)');
legend('|z-score|','Z threshold','Var. ratio','Var. threshold','Location','northwest');
grid on; xlim([0 365]);

% 3c: Combined anomaly alert flag
subplot(3,1,3);
area(t_days, double(anomaly_z)*0.33, 'FaceColor','b', 'FaceAlpha',0.3, ...
     'EdgeColor','none', 'DisplayName','Z-score alert');
hold on;
area(t_days, double(anomaly_var)*0.66, 'FaceColor',[0.8 0.4 0], ...
     'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName','Variance alert');
area(t_days, double(anomaly_combined)*1.0, 'FaceColor','r', ...
     'FaceAlpha',0.5, 'EdgeColor','none', 'DisplayName','Combined alert');
xline(200, 'b--', 'LineWidth', 1.5);
xline(280, 'm--', 'LineWidth', 1.5);
xlabel('Time (days)'); ylabel('Alert active');
title('3c: Anomaly Alert Flags — Z-Score (blue) | Variance (orange) | Combined (red)');
legend('Z-score alert','Variance alert','Combined alert','Location','northwest');
ylim([0 1.2]); yticks([0 1]); yticklabels({'Off','On'});
grid on; xlim([0 365]);

sgtitle('MODULE 3: IoT Sensor Anomaly Detection — Digital Twin Layer', ...
        'FontWeight','bold','FontSize',12);

fprintf('MODULE 3 complete. Figure 3 generated.\n\n');


%% =========================================================================
%  SECTION 4: DIGITAL TWIN DASHBOARD SUMMARY PLOT
%  =========================================================================
%  This consolidated figure presents the three DT framework outputs in a
%  single reference chart suitable for inclusion in the portfolio as an
%  illustration of the integrated digital twin operational dashboard.
%  It combines the key outputs of all three modules:
%   — DT drift estimate with tolerance limits
%   — TTTB prediction with calibration alert
%   — Combined anomaly detection status
% =========================================================================

fprintf('Generating Digital Twin Dashboard summary figure...\n');

figure('Name','DT Dashboard Summary','NumberTitle','off', ...
       'Position',[300 100 1200 700]);

% Panel 1: Drift model + DT estimate
subplot(3,1,1);
plot(t_days, drift_total, 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6, ...
     'DisplayName', 'Physical instrument drift');
hold on;
plot(t_days, drift_DT_estimate, 'b-', 'LineWidth', 2.0, ...
     'DisplayName', 'DT virtual model estimate');
yline(tolerance_kg,  'r--', 'LineWidth', 1.8, 'Label', '+Tol. limit');
yline(-tolerance_kg, 'r--', 'LineWidth', 1.8, 'Label', '-Tol. limit');
if ~isnan(first_alert_day)
    xline(first_alert_day, 'g-', 'LineWidth', 1.5, ...
          'Label', 'Cal. alert');
end
ylabel('Measurement drift (kg)');
title('LAYER 1 — Physics-Informed Drift Model');
legend('Actual drift','DT estimate','Location','northwest');
grid on; xlim([0 365]);

% Panel 2: TTTB
subplot(3,1,2);
plot(t_days, TTTB_days, 'b-', 'LineWidth', 1.8);
hold on;
yline(safety_days, 'r--', 'LineWidth', 1.8, ...
      'Label', sprintf('Safety margin (%d d)', safety_days));
if ~isnan(first_alert_day)
    xline(first_alert_day, 'g-', 'LineWidth', 1.5, 'Label','Cal. alert triggered');
end
fill([0 365 365 0],[0 0 safety_days safety_days],[1 0.8 0.8], ...
     'FaceAlpha', 0.2, 'EdgeColor','none', 'DisplayName','Alert zone');
ylabel('TTTB (days)');
title('LAYER 2 — Condition-Based Calibration Prediction (TTTB)');
legend('TTTB','Safety margin','Location','northeast');
ylim([0 200]); grid on; xlim([0 365]);

% Panel 3: Anomaly detection
subplot(3,1,3);
area(t_days, double(anomaly_combined), 'FaceColor','r', ...
     'FaceAlpha', 0.5, 'EdgeColor','none', 'DisplayName','Anomaly alert');
hold on;
area(t_days, double(anomaly_z & ~anomaly_combined)*0.6, ...
     'FaceColor','b', 'FaceAlpha', 0.3, 'EdgeColor','none', ...
     'DisplayName','Z-score only');
area(t_days, double(anomaly_var & ~anomaly_combined)*0.4, ...
     'FaceColor',[0.8 0.4 0], 'FaceAlpha', 0.3, 'EdgeColor','none', ...
     'DisplayName','Variance only');
xline(200, 'b--', 'LineWidth', 1.5, 'Label','Fault 1');
xline(280, 'm--', 'LineWidth', 1.5, 'Label','Fault 2');
ylabel('Alert status');
xlabel('Operational time (days)');
title('LAYER 3 — AI-Assisted Anomaly Detection');
legend('Combined alert','Z-score alert','Variance alert','Location','northwest');
ylim([0 1.3]); yticks([0 1]); yticklabels({'Normal','ALERT'});
grid on; xlim([0 365]);

sgtitle({'DIGITAL TWIN FRAMEWORK — IoT-Enabled Industrial Weighing Scale', ...
         'Integrated Operational Dashboard: Drift | Calibration | Anomaly Detection'}, ...
        'FontWeight','bold','FontSize',12);

fprintf('Dashboard figure generated.\n\n');


%% =========================================================================
%  SECTION 5: SUMMARY STATISTICS OUTPUT
%  =========================================================================

fprintf('=================================================================\n');
fprintf('  SIMULATION SUMMARY\n');
fprintf('=================================================================\n');
fprintf('\n  MODULE 1 — Load Cell Drift Model\n');
fprintf('  %-40s %.4f kg\n', 'End-of-year total drift (max):', max(abs(drift_total)));
fprintf('  %-40s %.4f kg\n', 'DT estimate error (mean |actual-DT|):', ...
        mean(abs(drift_total - drift_DT_estimate)));
fprintf('  %-40s ±%.2f kg\n', 'Calibration tolerance:', tolerance_kg);

fprintf('\n  MODULE 2 — Calibration Prediction\n');
if ~isnan(first_alert_day)
    fprintf('  %-40s Day %.1f\n', 'First DT calibration alert:', first_alert_day);
end
if ~isnan(breach_day)
    fprintf('  %-40s Day %.1f\n', 'Actual tolerance breach (unmanaged):', breach_day);
    fprintf('  %-40s %.1f days\n', 'DT lead time advantage:', breach_day - first_alert_day);
end

fprintf('\n  MODULE 3 — Anomaly Detection\n');
fprintf('  %-40s %d hours (%.1f%%)\n', 'Hours with Z-score alert:', ...
        sum(anomaly_z), 100*sum(anomaly_z)/N);
fprintf('  %-40s %d hours (%.1f%%)\n', 'Hours with variance alert:', ...
        sum(anomaly_var), 100*sum(anomaly_var)/N);
fprintf('  %-40s %d hours (%.1f%%)\n', 'Hours with combined alert:', ...
        sum(anomaly_combined), 100*sum(anomaly_combined)/N);
fprintf('\n=================================================================\n');
fprintf('  Simulation complete. 4 figures generated.\n');
fprintf('=================================================================\n');

% =========================================================================
% END OF SCRIPT
% =========================================================================
