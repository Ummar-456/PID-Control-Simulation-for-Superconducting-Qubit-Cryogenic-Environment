import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model Parameters
C_mxc = 0.5      # Thermal capacitance of MXC stage (J/K)
T0_mxc = 0.010   # Sink temperature / Base temperature MXC would cool to (K)
R_eff_mxc = 50.0 # Effective thermal resistance (K/W)

# Initial condition
T_initial_mxc = 0.010 # MXC starts at the sink temperature (K)

# --- Heater Power Profile (Open-Loop) ---
# Let's define a simple step in heater power
# P_heater will be a function of time: P_heater(t)
def heater_power(t):
    if t < 10:  # Seconds
        return 0.0  # No heater power initially
    elif t < 200: # Apply some power
        return 0.0002 # 0.2 mW = 200 uW
    else: # Turn off
        return 0.0

# --- Disturbance Load 
def disturbance_load(t):
    # Example: a small, constant background load or a step disturbance
    # if t > 300 and t < 400:
    #     return 0.00005 # 50 uW disturbance
    return 0.0 # No disturbance for now

# --- Define the Differential Equation for the Thermal Model ---
# dT/dt = (P_heater(t) + P_disturbance(t) - (T(t) - T0_mxc)/R_eff_mxc) / C_mxc
def mxc_thermal_model(t, T, C, T0, R_eff, P_h_func, P_d_func):
    P_h = P_h_func(t)
    P_d = P_d_func(t)
    dTdt = (P_h + P_d - (T - T0) / R_eff) / C
    return dTdt

# --- Simulation Time Span ---
t_start = 0
t_end = 600  # Simulate for 600 seconds (10 minutes)
t_eval = np.linspace(t_start, t_end, 1000) # Points at which to store the solution

# --- Run the Simulation (Open-Loop) ---
# solve_ivp requires arguments to the model to be passed via 'args'
sol = solve_ivp(
    mxc_thermal_model,
    [t_start, t_end],
    [T_initial_mxc],
    t_eval=t_eval,
    args=(C_mxc, T0_mxc, R_eff_mxc, heater_power, disturbance_load)
)

# Extract results
time = sol.t
temperature_mxc = sol.y[0]

# --- Plot the Results ---
plt.figure(figsize=(12, 8))

# Plot Temperature
plt.subplot(2, 1, 1)
plt.plot(time, temperature_mxc * 1000, label="MXC Temperature") # Plot in mK
plt.axhline(T0_mxc * 1000, color='gray', linestyle='--', label=f"T0 ({T0_mxc*1000:.1f} mK)")
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (mK)")
plt.title("Open-Loop Simulation of MXC Stage Temperature")
plt.legend()
plt.grid(True)

# Plot Heater Power 
P_h_values = [heater_power(t_i) for t_i in time]
P_d_values = [disturbance_load(t_i) for t_i in time]
plt.subplot(2, 1, 2)
plt.plot(time, np.array(P_h_values) * 1000, label="Heater Power (mW)", color='red')
# plt.plot(time, np.array(P_d_values) * 1000, label="Disturbance (mW)", color='orange', linestyle=':') # Uncomment to show disturbance
plt.xlabel("Time (seconds)")
plt.ylabel("Power (mW)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ---  Print final values ---
print(f"Initial temperature: {temperature_mxc[0]*1000:.2f} mK")
print(f"Temperature at t=10s (before heater): {temperature_mxc[np.searchsorted(time, 10)]*1000:.2f} mK")
print(f"Temperature at t=199s (heater on): {temperature_mxc[np.searchsorted(time, 199)]*1000:.2f} mK")
print(f"Final temperature: {temperature_mxc[-1]*1000:.2f} mK")

# Steady-state temperature calculation with P_heater = 0.0002 W
# dT/dt = 0  => P_heater - (T_ss - T0)/R_eff = 0
# T_ss = P_heater * R_eff + T0
P_h_steady = 0.0002 # 0.2 mW
T_ss_expected = (P_h_steady * R_eff_mxc + T0_mxc) * 1000 # in mK
print(f"Expected steady-state temperature with {P_h_steady*1000} mW heater: {T_ss_expected:.2f} mK")
