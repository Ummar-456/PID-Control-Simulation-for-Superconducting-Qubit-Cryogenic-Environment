# PID-Control-Simulation-for-Superconducting-Qubit-Cryogenic-Environment

# PID Control Simulation for Superconducting Qubit Cryogenic Environment

## 1. Introduction & Motivation

Quantum computers, particularly those based on superconducting qubits, rely on extremely stable, ultra-low temperature environments for operation. Qubits are highly sensitive to thermal fluctuations, which can introduce noise, limit coherence times, and increase gate errors, ultimately corrupting quantum computations. Achieving millikelvin-level temperature stability is therefore a critical hardware challenge.

This project, developed as a portfolio piece for graduate studies in quantum computing, bridges the gap between a classical Mechanical Engineering background and the physical requirements of quantum hardware. It explores the design and simulation of a **PID (Proportional-Integral-Derivative) feedback control system** to regulate the temperature of a cryostat's mixing chamber (MXC) stage, where superconducting qubits are typically mounted.

The goal is to demonstrate a complete engineering control process: from system modeling and controller design to comparing different tuning methodologies and critically analyzing their performance.

## 2. System Modeling

The thermal behavior of the MXC stage is approximated as a simplified first-order linear system. This model captures the essential dynamics for control system design without the complexity of full cryostat thermodynamics.

The system is described by the differential equation:

$$ C \frac{dT(t)}{dt} = P_{heater}(t) + P_{disturbance}(t) - \frac{T(t) - T_0}{R_{eff}} $$

Where:
- **$T(t)$**: Temperature of the MXC stage (K).
- **$C$**: Thermal capacitance of the stage (J/K).
- **$P_{heater}(t)$**: Power from the controllable heater (W). This is the controller's output.
- **$P_{disturbance}(t)$**: External heat loads acting as disturbances (W).
- **$T_0$**: The base temperature the stage would settle to without heater power (K).
- **$R_{eff}$**: An effective thermal resistance to the cooling source (K/W).

The simulation uses plausible parameters to model a realistic control challenge.

## 3. Controller Design & Tuning Methodologies

A PID controller was designed and implemented in Python to maintain a setpoint temperature of **15 mK**. Two distinct tuning methodologies were explored and compared.

### 3.1 Manual Tuning

An iterative, manual tuning process was used to find a stable set of PID gains. By individually adjusting the Proportional ($K_p$), Integral ($K_i$), and Derivative ($K_d$) terms, a controller was developed that successfully achieved the setpoint with minimal overshoot and effective disturbance rejection.

The plot below shows the performance of a well-tuned PI controller (`Kp=0.001, Ki=0.0008, Kd=0.0`), which eliminates steady-state error but shows some overshoot. Further manual tuning of the D-term can dampen these oscillations.

![1](https://github.com/user-attachments/assets/3defacbd-e860-4770-977d-95423f2daf94)


### 3.2 Ziegler-Nichols (Z-N) Method

To explore a formal, automated tuning approach, the classic Ziegler-Nichols Reaction Curve method was implemented.

1.  **Open-Loop Analysis:** An open-loop step test was simulated to get the system's reaction curve.
2.  **Parameter Extraction:** Key parameters—Process Gain ($K$), Time Delay ($L$), and Time Constant ($\tau$)—were programmatically extracted from the curve's maximum slope.
3.  **Gain Calculation:** These parameters were used with the Z-N tuning table to calculate the PID gains automatically.

The plot below shows the automated analysis of the reaction curve.

![2](https://github.com/user-attachments/assets/4af9f91b-e653-4c60-a08e-58bc2843953d)


## 4. Results & Critical Analysis

The PID gains calculated by the Ziegler-Nichols method were found to be `Kp=1.025, Ki=0.900, Kd=0.292`. When these gains were applied to the system, the controller was highly aggressive and unstable, resulting in sustained, high-frequency oscillations and heater "chattering."

![2](https://github.com/user-attachments/assets/651d545c-abc1-423b-a727-f8b98013f3a6)



This result, while producing an unstable controller, is a key success of the project. It demonstrates a critical engineering insight: heuristic tuning rules like Ziegler-Nichols can perform poorly on systems with a low time-delay-to-time-constant ratio ($\L/\tau$), which is characteristic of this thermal model. The resulting aggressive gains are unsuitable for a sensitive quantum application.

This comparison highlights that while formal methods provide a valuable framework, practical engineering judgment and iterative tuning are essential for designing robust control systems for real-world applications.

## 5. Conclusion

This project successfully demonstrates the application of classical control theory to a critical problem in quantum hardware engineering. By modeling a cryogenic system and comparing different PID tuning strategies, it showcases a deep understanding of dynamic systems, feedback control, and the importance of critical analysis in engineering design. This work affirms my passion for applying my engineering skills to solve the physical challenges inherent in building functional quantum computers.

## 6. How to Run

The entire process is contained within the `pid_cryo_simulation.py` script.

1.  Ensure you have Python installed with the required libraries: NumPy, SciPy, and Matplotlib.
    ```bash
    pip install numpy scipy matplotlib
    ```
2.  Run the script from your terminal:
    ```bash
    python pid_cryo_simulation.py
    ```
The script will first perform the open-loop analysis, show a plot, and then use the calculated Z-N gains to run and plot the closed-loop simulation. To test the manually tuned gains, you can modify the gain values in the "PART 2" section of the script.

## 7. Technologies Used
- Python 3
- NumPy
- SciPy (for `solve_ivp`)
- Matplotlib
