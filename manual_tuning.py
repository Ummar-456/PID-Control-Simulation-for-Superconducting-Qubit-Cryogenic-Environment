import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model Parameters (Illustrative) ---
C_mxc = 0.5      # Thermal capacitance of MXC stage (J/K)
T0_mxc = 0.010   # Sink temperature / Base temperature MXC would cool to (K)
R_eff_mxc = 50.0 # Effective thermal resistance (K/W)

# --- PID Controller Parameters ---
Kp = 0.001   # Keep this from the last run
Ki = 0.0008  # Add a small Integral gain
Kd = 0.002
# --- Simulation & Control Settings ---
T_setpoint = 0.015  # Desired MXC temperature (15 mK)
T_initial_mxc = 0.010 # Assume MXC starts at the sink temperature (K)

dt = 1.0      # Time step for PID controller and simulation (seconds)
sim_time = 600 # Total simulation time (seconds)

# --- Disturbance Load (Optional) ---
def disturbance_load(t):
    if t > 300 and t < 350: # Introduce a disturbance
        return 0.00005 # 50 uW disturbance
    return 0.0

# --- Define the Differential Equation for the Thermal Model ---
# dT/dt = (P_heater + P_disturbance(t) - (T - T0_mxc)/R_eff_mxc) / C_mxc
# Note: P_heater is now an argument, not a function of time directly in the model
def mxc_thermal_model_for_pid(t, T, C, T0, R_eff, P_h_current, P_d_func):
    P_d = P_d_func(t) # Disturbance can still be a function of time
    dTdt = (P_h_current + P_d - (T - T0) / R_eff) / C
    return dTdt

# --- PID Controller State Variables ---
integral_error = 0.0
previous_error = 0.0

# --- Data Logging ---
time_points = np.arange(0, sim_time, dt)
temperature_history = []
heater_power_history = []
error_history = []
setpoint_history = []

# --- Initial Temperature ---
current_T_mxc = T_initial_mxc

# --- Simulation Loop ---
print("Starting closed-loop PID simulation...")
for t_current in time_points:
    # Log current state
    temperature_history.append(current_T_mxc)
    setpoint_history.append(T_setpoint)

    # Calculate error
    error = T_setpoint - current_T_mxc
    error_history.append(error)

    # Proportional term
    P_term = Kp * error

    # Integral term
    integral_error += error * dt
    I_term = Ki * integral_error
    # Anti-windup (optional but good practice for Ki > 0)
    # For example, clamp integral term if heater power maxes out
    # (We'll add heater limits later if needed)

    # Derivative term
    derivative_error = (error - previous_error) / dt
    D_term = Kd * derivative_error
    previous_error = error

    # PID controller output (heater power)
    P_heater_current = P_term + I_term + D_term

    # Apply limits to heater power (e.g., cannot be negative, has a max)
    P_heater_current = max(0, P_heater_current) # No negative heating
    P_heater_max = 0.001 # e.g., 1 mW max heater power
    P_heater_current = min(P_heater_current, P_heater_max)
    
    heater_power_history.append(P_heater_current)

    # Simulate one time step for the MXC stage
    # We need to pass the current P_heater, not a function
    sol_step = solve_ivp(
        mxc_thermal_model_for_pid,
        [0, dt], # Simulate for one dt
        [current_T_mxc],
        t_eval=[dt], # Evaluate only at the end of the step
        args=(C_mxc, T0_mxc, R_eff_mxc, P_heater_current, lambda t_inner: disturbance_load(t_current + t_inner)) # Pass current time for disturbance
    )
    current_T_mxc = sol_step.y[0, -1]

# Convert lists to numpy arrays for plotting
temperature_history = np.array(temperature_history)
heater_power_history = np.array(heater_power_history)
error_history = np.array(error_history)
setpoint_history = np.array(setpoint_history)

# --- Plot the Results ---
plt.figure(figsize=(12, 10))

# Plot Temperature and Setpoint
plt.subplot(3, 1, 1)
plt.plot(time_points, temperature_history * 1000, label="MXC Temperature (mK)")
plt.plot(time_points, setpoint_history * 1000, label="Setpoint (mK)", color='r', linestyle='--')
plt.axhline(T0_mxc * 1000, color='gray', linestyle=':', label=f"T0 ({T0_mxc*1000:.1f} mK)")
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (mK)")
plt.title(f"PID Control of MXC Temperature (Kp={Kp}, Ki={Ki}, Kd={Kd})")
plt.legend()
plt.grid(True)

# Plot Heater Power
plt.subplot(3, 1, 2)
plt.plot(time_points, heater_power_history * 1000, label="Heater Power (mW)", color='green')
plt.xlabel("Time (seconds)")
plt.ylabel("Power (mW)")
plt.legend()
plt.grid(True)

# Plot Error
plt.subplot(3, 1, 3)
plt.plot(time_points, error_history * 1000, label="Error (mK)", color='purple')
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("Error (mK)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Setpoint: {T_setpoint*1000:.2f} mK")
print(f"Final temperature: {temperature_history[-1]*1000:.2f} mK")
print(f"Final error: {error_history[-1]*1000:.3f} mK")
