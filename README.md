# 🤖 ICRL-Agent
*A Deep Reinforcement Learning Agent with Intermittent Control for Autonomous Driving*

---

## 🧭 Overview
**ICRL-Agent** is a research project developed as part of my MSc thesis,  
**“Deep Learning Solutions for Autonomous Vehicles: Investigating the Impact of Intermittent Control on Reinforcement Learning.”**

The project explores how **intermittent control theory**, inspired by human decision-making and motor control, can improve the **stability**, **efficiency**, and **robustness** of deep reinforcement learning (RL) agents operating in autonomous driving environments.

Two main environments were used for experimentation:
- **CARLA Simulator** — realistic autonomous vehicle training setup.  
- **Custom Maze Solver** — simplified RL environment for controlled algorithmic testing.

---

## 🧱 Repository Structure

| File | Description |
|------|--------------|
| `av_carla_icrl_agent.py` | Deep Q-Network (DQN) implementation with intermittent control for CARLA autonomous driving. |
| `av_carla_icrl_agent_demo.mp4` | Demonstration of the trained CARLA ICRL agent. |
| `maze_solver_icrl_agent.py` | Intermittent-control DQN applied to a 2D maze navigation task. |
| `maze_solver_dqn_agent.ipynb` | Baseline DQN maze-solving model (without intermittent control). |
| `maze_solver_qlearning_agent.ipynb` | Classical Q-Learning version for performance comparison. |

---

## ⚙️ Methodology
The research investigates how introducing **decision intervals** (intermittent control) affects RL agent behavior.

**Key Steps:**
1. Implemented **Deep Q-Networks (DQN)** with discrete control timing.  
2. Integrated **intermittent control logic** to simulate human-like reaction delays.  
3. Trained and compared **continuous vs. intermittent** control policies.  
4. Evaluated metrics focused on **training efficiency** rather than control accuracy,  
   including:
   - **Convergence speed** — time required to reach reward stabilization.  
   - **Sample efficiency** — number of episodes needed for consistent performance.  
   - **Computation cost** — training time per episode and overall runtime.  
   - **Stability under intermittent updates** — observing variance in learning curves.  

---

## 🧪 Notes
The experiments demonstrated that intermittent control can reduce the overall training
time required for convergence while maintaining comparable reward performance to
continuously controlled agents — suggesting a potential improvement in computational
efficiency for resource-constrained RL systems.

- Designed for **academic research** and proof-of-concept evaluation.  
- Code focuses on demonstrating behavior differences, not production optimization.  
- Results may vary depending on **TensorFlow/PyTorch** and **CARLA** versions.  

---

## 📊 Results

Experiments were carried out in two environments — a **maze simulator** and the **CARLA autonomous driving simulator** — to evaluate the impact of **Intermittent Control (IC)** on Deep Q-Network (DQN) performance.

- **Maze environment:**  
  The IC factor of 6 achieved **16× faster execution** and required **5× fewer episodes** to complete training compared to the baseline DQN, while maintaining identical task performance.  
  These results demonstrate that IC significantly accelerates learning in environments where training is dominated by agent–environment interactions.

- **CARLA environment:**  
  The IC factor of 20 resulted in **13.5× fewer network updates** and **22× lower loss**, while maintaining a **90% success rate**.  
  Because CARLA runs in *real time*, execution time is bound by the simulator’s frame rate.  
  Consequently, although the algorithm was **computationally more efficient**, total *wall-clock execution time* increased slightly (~25%) due to longer episode durations.  
  This indicates that IC improves learning efficiency and stability but provides limited runtime gains in real-time simulators.

Overall, intermittent control substantially improved **training efficiency**, **stability**, and **resource utilization** without compromising performance.  
Its effects were most pronounced in simulated environments where computation time dominates over real-time constraints.

---

## 🧰 Technologies Used
Python • TensorFlow • NumPy • OpenAI Gym • CARLA Simulator • Jupyter  

---

## 🎓 Academic Context
This repository supports my MSc thesis submitted to  
**Manchester Metropolitan University (2020)**.  

The research investigates how integrating intermittent control theory into RL can enhance decision-making reliability and performance for autonomous vehicles.

---

## 🎥 Demo
<p align="center">
  <a href="av_carla_icrl_agent_demo.mp4">▶️ Watch CARLA Driving Demo</a>
</p>
