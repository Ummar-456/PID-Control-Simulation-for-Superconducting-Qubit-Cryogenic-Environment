import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# ==============================================================================
# PART 1: DEFINING THE SIMULATION & COST FUNCTION
# ==============================================================================

# --- Model Parameters (Constant for all simulations) ---
C_mxc = 0.5      # Thermal capacitance of MXC stage (J/K)
T0_mxc = 0.010   # Sink temperature / Base temperature (K)
R_eff_mxc = 50.0 # Effective thermal resistance (K/W)
T_initial_mxc = 0.010 # Starting temperature (K)

# --- Simulation Settings (Constant) ---
T_setpoint = 0.015  # Desired MXC temperature (15 mK)
dt = 1.0            # Time step for PID controller (s)
sim_time = 600      # Total simulation time (s)

# --- Disturbance Load ---
def disturbance_load(t):
    if t > 300 and t < 350: return 0.00005 # 50 uW disturbance
    return 0.0

# --- Noise Settings ---
SENSOR_NOISE_STD_DEV = 0.00005 # 0.05 mK standard deviation on sensor reading

# --- Unified Thermal Model (Corrected) ---
def mxc_thermal_model_pid(t, T, C, T0, R_eff, P_h_current, P_d_current):
    """
    Thermal model where heater and disturbance power are given as fixed numbers
    for the duration of the integration step.
    """
    dTdt = (P_h_current + P_d_current - (T - T0) / R_eff) / C
    return dTdt

# --- The core simulation function to be optimized ---
def simulate_pid_and_get_cost(gains):
    """
    Runs a full PID simulation for a given set of gains (Kp, Ki, Kd)
    and returns a single cost value (ITAE) representing performance.
    """
    Kp, Ki, Kd = gains
    
    integral_error, previous_error = 0.0, 0.0
    time_points = np.arange(0, sim_time, dt)
    error_history = []
    current_T_mxc = T_initial_mxc

    for t_current in time_points:
        T_measured = current_T_mxc + np.random.normal(0, SENSOR_NOISE_STD_DEV)
        error = T_setpoint - T_measured
        error_history.append(error)

        integral_error += error * dt
        derivative_error = (error - previous_error) / dt
        previous_error = error
        
        P_heater_current = Kp * error + Ki * integral_error + Kd * derivative_error
        P_heater_current = max(0, min(P_heater_current, 0.001))

        current_disturbance = disturbance_load(t_current)
        sol_step = solve_ivp(
            mxc_thermal_model_pid, [0, dt], [current_T_mxc], t_eval=[dt],
            args=(C_mxc, T0_mxc, R_eff_mxc, P_heater_current, current_disturbance)
        )
        current_T_mxc = sol_step.y[0, -1]

    cost = np.sum(np.arange(0, sim_time, dt) * np.abs(np.array(error_history)) * dt)
    return cost

# ==============================================================================
# PART 2: RUNNING THE OPTIMIZATION
# ==============================================================================
print("--- Starting Part 2: Auto-Tuning PID Gains with Optimizer ---")

initial_gains = [0.001, 0.0008, 0.002]
bounds = ((0, None), (0, None), (0, None))

start_time = time.time()
result = minimize(
    simulate_pid_and_get_cost, initial_gains, method='Nelder-Mead',
    bounds=bounds, options={'xatol': 1e-5, 'fatol': 1e-3, 'disp': True}
)
end_time = time.time()
print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

optimal_gains = result.x
Kp_opt, Ki_opt, Kd_opt = optimal_gains

print("\n--- Optimal Gains Found by Auto-Tuner ---")
print(f"Optimal Kp = {Kp_opt:.4f}")
print(f"Optimal Ki = {Ki_opt:.4f}")
print(f"Optimal Kd = {Kd_opt:.4f}")

# ==============================================================================
# PART 3: VISUALIZE THE PERFORMANCE OF THE OPTIMAL CONTROLLER
# ==============================================================================
print("\n--- Starting Part 3: Simulating with Optimal Gains ---")

# Re-run the simulation one last time with the optimal gains to log data for plotting.
integral_error, previous_error = 0.0, 0.0
time_points = np.arange(0, sim_time, dt)
temperature_history, heater_power_history, error_history = [], [], []
setpoint_history = []
current_T_mxc = T_initial_mxc

for t_current in time_points:
    temperature_history.append(current_T_mxc)
    setpoint_history.append(T_setpoint)
    T_measured = current_T_mxc + np.random.normal(0, SENSOR_NOISE_STD_DEV)
    error = T_setpoint - T_measured
    error_history.append(error)
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    previous_error = error
    P_heater_current = Kp_opt * error + Ki_opt * integral_error + Kd_opt * derivative_error
    P_heater_current = max(0, min(P_heater_current, 0.001))
    heater_power_history.append(P_heater_current)
    current_disturbance = disturbance_load(t_current)
    sol_step = solve_ivp(
        mxc_thermal_model_pid, [0, dt], [current_T_mxc], t_eval=[dt],
        args=(C_mxc, T0_mxc, R_eff_mxc, P_heater_current, current_disturbance)
    )
    current_T_mxc = sol_step.y[0, -1]

# --- Plot the Final Results ---
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(time_points, np.array(temperature_history) * 1000, label="MXC Temperature (mK)")
plt.plot(time_points, np.array(setpoint_history) * 1000, label="Setpoint (mK)", color='r', linestyle='--')
plt.title(f"Auto-Tuned PID Control (Kp={Kp_opt:.3f}, Ki={Ki_opt:.3f}, Kd={Kd_opt:.3f})")
plt.ylabel("Temperature (mK)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_points, np.array(heater_power_history) * 1000, label="Heater Power (mW)", color='green')
plt.ylabel("Power (mW)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_points, np.array(error_history) * 1000, label="Error (mK)", color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("Error (mK)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
