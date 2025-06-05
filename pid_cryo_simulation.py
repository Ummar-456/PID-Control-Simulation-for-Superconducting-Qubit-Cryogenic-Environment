# ==============================================================================
# PID Control Simulation for Superconducting Qubit Cryogenic Environment
#
# This script demonstrates a full control system design process:
# 1. Open-Loop Analysis of a thermal model for a cryostat stage.
# 2. Programmatic calculation of PID gains using the Ziegler-Nichols method.
# 3. Closed-Loop Simulation of the system using the calculated gains.
#
# Author: [Your Name]
# Date: June 5, 2025
# ==============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model Parameters (Illustrative) ---
# These parameters model the thermal behavior of a cryostat's Mixing Chamber (MXC) stage
C_mxc = 0.5      # Thermal capacitance of MXC stage (J/K)
T0_mxc = 0.010   # Sink temperature / Base temp MXC would cool to without heating (K)
R_eff_mxc = 50.0 # Effective thermal resistance to the cooling sink (K/W)
T_initial_mxc = 0.010 # Assume MXC starts at the sink temperature (K)

# ==============================================================================
# PART 1: OPEN-LOOP ANALYSIS for Ziegler-Nichols Tuning
# ==============================================================================
print("--- Starting Part 1: Open-Loop Analysis for Z-N Tuning ---")

# --- Step Test Power Profile ---
P_step_start_time = 50.0  # Time the heater turns on
P_step_power = 0.0002     # Heater power for the step test (0.2 mW)

def open_loop_heater_power(t):
    """Defines a step input for heater power."""
    if t >= P_step_start_time:
        return P_step_power
    return 0.0

# --- Define the Differential Equation for the Thermal Model ---
def mxc_thermal_model(t, T, C, T0, R_eff, P_h_func):
    """First-order thermal model of the cryostat stage."""
    P_h = P_h_func(t)
    dTdt = (P_h - (T - T0) / R_eff) / C
    return dTdt

# --- Run the Open-Loop Simulation ---
t_sim_end = 600
# Use a higher number of evaluation points for a smoother curve and more accurate gradient
t_eval = np.linspace(0, t_sim_end, 2000) 
sol_open_loop = solve_ivp(
    mxc_thermal_model, [0, t_sim_end], [T_initial_mxc], t_eval=t_eval,
    args=(C_mxc, T0_mxc, R_eff_mxc, open_loop_heater_power)
)
time_ol = sol_open_loop.t
temp_ol = sol_open_loop.y[0]

# --- Programmatically Calculate K, L, and tau for Ziegler-Nichols ---

# 1. Calculate Process Gain (K)
# This measures the steady-state change in temperature for a change in power
T_ss_initial = temp_ol[0] 
T_ss_final = temp_ol[-1] 
K_process_gain = (T_ss_final - T_ss_initial) / P_step_power
print(f"Calculated Process Gain (K): {K_process_gain:.2f} K/W")

# 2. Find Maximum Slope and Tangent Line
# Calculate the derivative (slope) of the temperature curve
temp_slope = np.gradient(temp_ol, time_ol)
max_slope_index = np.argmax(temp_slope)
max_slope = temp_slope[max_slope_index]

# Get the time and temperature at the point of max slope
t_max_slope = time_ol[max_slope_index]
T_max_slope = temp_ol[max_slope_index]
print(f"Max slope of {max_slope*1000:.4f} mK/s found at t={t_max_slope:.2f} s")

# 3. Calculate Time Delay (L)
# Find time 't' where tangent line intersects the initial temperature line
# Equation: T_initial = max_slope * (t - t_max_slope) + T_max_slope
t_intersect_initial = (T_ss_initial - T_max_slope) / max_slope + t_max_slope
L_time_delay = t_intersect_initial - P_step_start_time
print(f"Calculated Time Delay (L): {L_time_delay:.2f} s")

# 4. Calculate Time Constant (tau)
# Find time 't' where tangent line intersects the final temperature line
t_intersect_final = (T_ss_final - T_max_slope) / max_slope + t_max_slope
tau_time_constant = t_intersect_final - t_intersect_initial
print(f"Calculated Time Constant (tau): {tau_time_constant:.2f} s")


# --- Plot the Open-Loop Analysis ---
plt.figure(figsize=(10, 6))
plt.plot(time_ol, temp_ol * 1000, label='System Response (mK)')
# Plot the tangent line for visualization
tangent_line_t = np.array([t_intersect_initial, t_intersect_final])
tangent_line_T = (tangent_line_t - t_max_slope) * max_slope + T_max_slope
plt.plot(tangent_line_t, tangent_line_T * 1000, 'r--', label='Tangent at Max Slope')
plt.axhline(T_ss_initial * 1000, color='gray', linestyle=':')
plt.axhline(T_ss_final * 1000, color='gray', linestyle=':')
plt.title("Ziegler-Nichols Reaction Curve Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (mK)")
plt.legend()
plt.grid(True)
plt.show(block=True)


# ==============================================================================
# PART 2: CLOSED-LOOP SIMULATION with Calculated Z-N Gains
# ==============================================================================
print("\n--- Starting Part 2: Closed-Loop Simulation with Z-N Gains ---")

# --- Calculate PID Gains using the Ziegler-Nichols Table ---
if L_time_delay > 1e-6: # Avoid division by zero if L is negligible
    # Classic PID Tuning Rules
    Kp_zn = 1.2 * tau_time_constant / (K_process_gain * L_time_delay)
    Ti_zn = 2 * L_time_delay   # Integral Time
    Td_zn = 0.5 * L_time_delay # Derivative Time
    
    Ki_zn = Kp_zn / Ti_zn # Convert from Ti to Ki
    Kd_zn = Kp_zn * Td_zn # Convert from Td to Kd
else:
    print("Time delay L is negligible. Z-N method is not suitable. Using default gains.")
    # Fallback to a manually tuned set of gains if L is too small
    Kp_zn, Ki_zn, Kd_zn = 0.002, 0.001, 0.0001 

print("\nCalculated Ziegler-Nichols PID Gains:")
print(f"Kp = {Kp_zn:.4f}")
print(f"Ki = {Ki_zn:.4f}")
print(f"Kd = {Kd_zn:.4f}")
print("(Note: These gains are often aggressive and may lead to instability)")

# --- Set which gains to use for the closed-loop simulation ---
USE_ZN_GAINS = True # Set to False to use manual gains below
if USE_ZN_GAINS:
    Kp, Ki, Kd = Kp_zn, Ki_zn, Kd_zn
else:
    # A manually tuned set of gains for comparison
    Kp, Ki, Kd = 0.001, 0.0008, 0.002 

# --- Simulation & Control Settings ---
T_setpoint = 0.015
dt = 1.0
sim_time = 600

# --- Disturbance Load ---
def disturbance_load(t):
    if t > 300 and t < 350: return 0.00005 # 50 uW disturbance
    return 0.0

# --- PID model and state variables ---
def mxc_thermal_model_for_pid(t, T, C, T0, R_eff, P_h_current, P_d_func):
    P_d = P_d_func(t)
    dTdt = (P_h_current + P_d - (T - T0) / R_eff) / C
    return dTdt

integral_error, previous_error = 0.0, 0.0
time_points = np.arange(0, sim_time, dt)
temperature_history, heater_power_history, error_history = [], [], []
setpoint_history = []
current_T_mxc = T_initial_mxc

# --- Simulation Loop ---
for t_current in time_points:
    # Log current state
    temperature_history.append(current_T_mxc)
    setpoint_history.append(T_setpoint)

    # Calculate PID terms
    error = T_setpoint - current_T_mxc
    error_history.append(error)
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    previous_error = error
    
    # Calculate controller output
    P_heater_current = Kp * error + Ki * integral_error + Kd * derivative_error
    
    # Apply actuator limits (heater can't be negative, has a max power)
    P_heater_current = max(0, min(P_heater_current, 0.001)) # 0 to 1 mW range
    heater_power_history.append(P_heater_current)

    # Simulate one time step for the MXC stage
    current_disturbance = disturbance_load(t_current) # Disturbance is constant over dt
    sol_step = solve_ivp(
        mxc_thermal_model_for_pid, [0, dt], [current_T_mxc], t_eval=[dt],
        args=(C_mxc, T0_mxc, R_eff_mxc, P_heater_current, lambda t: current_disturbance)
    )
    current_T_mxc = sol_step.y[0, -1]

# --- Plot the Closed-Loop Results ---
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(time_points, np.array(temperature_history) * 1000, label="MXC Temperature (mK)")
plt.plot(time_points, np.array(setpoint_history) * 1000, label="Setpoint (mK)", color='r', linestyle='--')
plt.title(f"PID Control (Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f})")
plt.ylabel("Temperature (mK)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_points, np.array(heater_power_history) * 1000, label="Heater Power (mW)", color='green')
plt.ylabel("Power (mW)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_points, np.array(error_history) * 1000, label="Error (mK)", color='purple')
plt.axhline(0, color='black', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("Error (mK)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
