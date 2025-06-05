# PID Control Simulation for Superconducting Qubit Cryogenic Environment

**Author:** Ummar Ahmed 
**Date:** June 2025
**Project for:** Quantum Computing and control

---

## 1. Overview & Motivation

The performance of leading quantum computing platforms, particularly those based on superconducting qubits, is critically dependent on maintaining an extremely stable, ultra-low temperature environment. Qubits are highly sensitive to thermal fluctuations, which can cause decoherence, increase gate error rates, and ultimately corrupt quantum computations. Achieving and maintaining millikelvin-level temperature stability inside a dilution refrigerator is therefore a crucial real-world engineering challenge.

This project bridges the gap between a classical Mechanical Engineering background and the physical requirements of quantum hardware. It demonstrates the design, simulation, and analysis of a **PID (Proportional-Integral-Derivative) feedback control system** to regulate the temperature of a cryostat's mixing chamber (MXC) stage.

The project showcases a complete engineering workflow:
- **System Modeling:** Creating a mathematical model of the physical system.
- **Controller Design:** Implementing a PID controller.
- **Tuning & Analysis:** Comparing multiple tuning strategies, from manual iteration to formal heuristics and computational optimization.
- **Realistic Simulation:** Incorporating elements like external disturbances and sensor noise to test controller robustness.

## 2. System Modeling

The thermal behavior of the MXC stage is approximated by a simplified first-order linear system. This model captures the essential dynamics needed for control system design.

The system is described by the differential equation:

$$ C \frac{dT(t)}{dt} = P_{heater}(t) + P_{disturbance}(t) - \frac{T(t) - T_0}{R_{eff}} $$

Where:
- **$T(t)$**: Temperature of the MXC stage (K).
- **$C$**: Thermal capacitance of the stage (J/K).
- **$P_{heater}(t)$**: Power from the controllable heater (W). This is the controller's output.
- **$P_{disturbance}(t)$**: External heat loads acting as disturbances (W).
- **$T_0$**: The base temperature the stage would settle to without heater power (K).
- **$R_{eff}$**: An effective thermal resistance to the cooling source (K/W).

The simulation uses plausible parameters to model a realistic control challenge with a target setpoint of **15 mK**.

## 3. Controller Design & Tuning Methodologies

A PID controller was designed and implemented in Python to maintain the setpoint temperature. Three distinct tuning approaches were investigated to find the optimal controller gains ($K_p, K_i, K_d$).

### 3.1 Manual Tuning

### 3.1 Manual Tuning

An iterative, manual tuning process was used to understand the effect of each PID term. This involved first tuning the Proportional ($K_p$) for a fast response, then adding the Integral ($K_i$) to eliminate steady-state error, and finally adding the Derivative ($K_d$) to dampen oscillations and reduce overshoot. This practical approach resulted in a stable and functional controller.

*<p align="center">Figure 1: A well-behaved controller achieved through careful manual tuning of PID gains.</p>*
<p align="center">
  <img src="https://github.com/user-attachments/assets/404ca5cb-b129-4cb8-a7e9-e910981616e5" width="800">
</p>

### 3.2 Ziegler-Nichols (Z-N) Method

### 3.2 Ziegler-Nichols (Z-N) Method

To explore a formal heuristic approach, the classic Ziegler-Nichols Reaction Curve method was programmatically implemented. The script first performed an open-loop step test, then automatically analyzed the resulting curve to extract the system's process gain ($K$), time delay ($L$), and time constant ($\tau$).

*<p align="center">Figure 2: Automated analysis of the open-loop step response to find Z-N parameters.</p>*
<p align="center">
  <img src="https://github.com/user-attachments/assets/1323e75e-8c20-475f-9b3f-b3c74744c59f" width="600">
</p>

These parameters were then used to calculate the PID gains according to the Z-N tuning table.

These parameters were then used to calculate the PID gains according to the Z-N tuning table.

### 3.3 Computational Optimization (Auto-Tuning)

As the most advanced approach, an "auto-tuner" was developed using the `scipy.optimize.minimize` function. A cost function based on the **Integral of Time-multiplied Absolute Error (ITAE)** was defined to numerically score the controller's performance. The optimizer's goal was to find the PID gains that minimized this cost, thereby automatically finding a high-performance tuning. To add realism, random noise was added to the simulated temperature sensor for this process.

## 4. Results & Critical Analysis

## 4. Results & Critical Analysis

The three tuning methods yielded dramatically different results, providing key insights into control system design.

The Ziegler-Nichols method produced highly aggressive gains (`Kp=1.025, Ki=0.900, Kd=0.292`), resulting in an **unstable controller** with severe oscillations and heater "chattering," as seen below.

*<p align="center">Figure 3: The unstable and oscillatory response from the Z-N tuned controller.</p>*
<p align="center">
  <img src="https://github.com/user-attachments/assets/0876d750-2f18-4335-9e88-56ddd9d9a545" width="800">
</p>

This outcome demonstrates a limitation of the Z-N heuristic for systems with a low time-delay-to-time-constant ratio. In contrast, the **auto-tuner successfully found optimal gains (`Kp=0.001, Ki=0.0008, Kd=0.0021`)** that produced an excellent response.

*<p align="center">Figure 4: The high-performance response from the auto-tuned PID controller.</p>*
<p align="center">
  <img src="https://github.com/user-attachments/assets/e0ec7d2a-9f22-4e8f-9596-3fce3beb9e6a" width="800">
</p>

The auto-tuned controller is clearly superior. It exhibits a fast response, minimal overshoot, quick settling time, and robustly rejects disturbances, even in the presence of sensor noise.


The auto-tuned controller is clearly superior. It exhibits a fast response, minimal overshoot, quick settling time, and robustly rejects disturbances, even in the presence of sensor noise.

## 5. Conclusion

This project successfully demonstrates the application of classical control theory to a critical problem in quantum hardware engineering. It highlights that while formal tuning methods are valuable, a deep understanding of the system and the limitations of these methods is required. The success of the computational optimization approach showcases a modern, powerful technique for designing robust controllers for complex systems.

This work solidifies my understanding of the practical engineering challenges in quantum computing and affirms my passion for applying my skills to help solve them.

## 6. How to Run

The final script containing the auto-tuner and simulation is `pid_cryo_simulation.py`.

1.  Ensure you have Python installed with the required libraries: NumPy, SciPy, and Matplotlib.
    ```bash
    pip install numpy scipy matplotlib
    ```
2.  Run the script from your terminal:
    ```bash
    python pid_cryo_simulation.py
    ```
The script will first run the optimization process (which may take a minute) and print the optimal gains it finds. It will then automatically run a final simulation with these gains and display the resulting performance plot.

## 7. Technologies Used
- **Language:** Python 3
- **Libraries:**
  - NumPy
  - SciPy (`solve_ivp` for ODE simulation, `minimize` for optimization)
  - Matplotlib (for plotting)
