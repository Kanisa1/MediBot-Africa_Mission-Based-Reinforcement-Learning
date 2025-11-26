

# **Medical Drone Delivery Using Reinforcement Learning**

*A Deep Reinforcement Learning Framework for Autonomous Medical Drone Navigation in Healthcare Environments*

---

## 📌 **Project Overview**

This project implements and compares multiple **Deep Reinforcement Learning (DRL)** algorithms to train an autonomous **medical supply delivery drone**. The environment simulates navigation tasks across three progressively complex phases:

* **Phase 1:** Simple navigation
* **Phase 2:** Obstacles + longer routes
* **Phase 3:** Constrained narrow corridors (hard)

The goal is to optimize the drone’s ability to deliver supplies safely, efficiently, and reliably under varying difficulty levels.

We compare four DRL algorithms:

* **PPO** (Proximal Policy Optimization)
* **A2C** (Advantage Actor-Critic)
* **DQN** (Deep Q-Network)
* **REINFORCE** (Policy Gradient w/ Baseline)

---

## ⚙️ **Environment Setup**

### **State Space**

The state includes:

* Drone’s current coordinates
* Target coordinates
* Distance to obstacles
* Phase difficulty indicators
* Collision risk indicators

### **Action Space**

Discrete motions:

* `Move_Forward`
* `Move_Backward`
* `Move_Left`
* `Move_Right`
* `Ascend`
* `Descend`
* `Hover`

### **Reward Function**

Reward shaping encourages:

✔ Efficient path planning
✔ Collision avoidance
✔ Stable movement
✔ Reaching the final target

Penalties include:

✖ Collisions
✖ Excessive path length
✖ Unstable/jerky motion

---

# 🧠 **Algorithms Implemented**

## **1. DQN (Deep Q-Network)**

* Off-policy value-based method
* Uses replay buffer + target networks
* Struggles in high-dimensional continuous navigation tasks

## **2. PPO (Proximal Policy Optimization)**

* On-policy actor-critic
* Clipped surrogate objective stabilizes learning
* Best performance in all metrics

## **3. A2C (Advantage Actor-Critic)**

* Parallelized policy learning
* Faster updates but higher variance

## **4. REINFORCE (Policy Gradient)**

* Pure policy gradient
* High variance, slow convergence
* Weakest performing algorithm

---

# 🚀 **Implementation**

Below are all hyperparameter comparisons for transparency and reproducibility.

---

## **🔧 DQN Hyperparameters & Results**

| Run     | Learning Rate | Gamma | Replay Buffer | Batch Size | Exploration | Mean Reward | Episodes to Converge |
| ------- | ------------- | ----- | ------------- | ---------- | ----------- | ----------- | -------------------- |
| 1       | 1e-4          | 0.99  | 100K          | 32         | ε-greedy    | 5,847       | ~450                 |
| 2       | 1e-4          | 0.99  | 100K          | 32         | ε-greedy    | 6,124       | ~420                 |
| 3       | 1e-4          | 0.99  | 100K          | 32         | ε-greedy    | 5,932       | ~480                 |
| 4       | 1e-4          | 0.99  | 100K          | 32         | ε-greedy    | 6,215       | ~410                 |
| …       | …             | …     | …             | …          | …           | …           | …                    |
| **Avg** | —             | —     | —             | —          | —           | **≈ 6,050** | **≈ 450**            |

---

## **🔧 PPO Hyperparameters & Results**

| Run     | LR     | n_steps | Batch | Epochs | Entropy | Mean Reward | Episodes to Converge |
| ------- | ------ | ------- | ----- | ------ | ------- | ----------- | -------------------- |
| 1       | 2.5e-4 | 2048    | 128   | 15     | 0.01    | 8,234       | ~280                 |
| 2       | 2.5e-4 | 2048    | 128   | 15     | 0.01    | 8,567       | ~270                 |
| 3       | 2.5e-4 | 2048    | 128   | 15     | 0.01    | 8,012       | ~300                 |
| …       | …      | …       | …     | …      | …       | …           | …                    |
| **Avg** | —      | —       | —     | —      | —       | **≈ 8,500** | **≈ 275**            |

---

## **🔧 A2C Hyperparameters & Results**

| Run     | LR   | n_steps | Batch | Entropy | ValueCoef | Mean Reward | Episodes to Converge |
| ------- | ---- | ------- | ----- | ------- | --------- | ----------- | -------------------- |
| 1       | 7e-4 | 16      | 16    | 0.01    | 0.5       | 7,123       | ~350                 |
| 2       | 7e-4 | 16      | 16    | 0.01    | 0.5       | 7,456       | ~320                 |
| 3       | 7e-4 | 16      | 16    | 0.01    | 0.5       | 6,987       | ~380                 |
| …       | …    | …       | …     | …       | …         | …           |                      |
| **Avg** | —    | —       | —     | —       | —         | **≈ 7,400** | **≈ 335**            |

---

## **🔧 REINFORCE Hyperparameters & Results**

| Run     | LR   | Baseline    | Entropy | Mean Reward | Episodes to Converge |
| ------- | ---- | ----------- | ------- | ----------- | -------------------- |
| 1       | 1e-3 | State Value | 0.01    | 3,456       | ~600                 |
| 2       | 1e-3 | State Value | 0.01    | 3,234       | ~650                 |
| 3       | 1e-3 | State Value | 0.01    | 3,678       | ~580                 |
| …       | …    | …           | …       | …           | …                    |
| **Avg** | —    | —           | —       | **≈ 3,500** | **≈ 600**            |

---

# 📊 **Results & Analysis**

## **1. Cumulative Reward Comparison**

| Algorithm     | Avg Reward | Rank      |
| ------------- | ---------- | --------- |
| **PPO**       | **~8,500** | 🥇 Best   |
| **A2C**       | ~7,400     | 🥈        |
| **DQN**       | ~6,050     | 🥉        |
| **REINFORCE** | ~3,500     | ❌ Weakest |

### Key Insights

✔ PPO produced the **highest and most stable** rewards
✔ A2C performed well but had higher variance
✔ DQN struggled with sample efficiency
✔ REINFORCE suffered from high-variance gradients

---

## **2. Phase-Specific Best Rewards**

| Phase            | Best Reward | Algorithm                |
| ---------------- | ----------- | ------------------------ |
| Phase 1 — Easy   | **13,101**  | PPO                      |
| Phase 2 — Medium | **10,851**  | PPO                      |
| Phase 3 — Hard   | **-169.84** | All algorithms struggled |

🔎 **Observation:**
Phase 3’s difficulty spike implies curriculum learning requires more smoothing.

---

# 📈 **Training Stability**

## **PPO — Most Stable (Score: 8.5/10)**

* Smooth surrogate objective
* Healthy entropy values
* Minimal oscillations

## **A2C — Moderately Stable (7.5/10)**

* Occasional spikes due to small n_steps
* Stable critic in most phases

## **DQN — Less Stable (6.5/10)**

* TD-error oscillations
* Sensitive to ε-decay schedule

---

# ⏱ **Convergence Speed**

| Algorithm | Avg Episodes to Converge | Rank       |
| --------- | ------------------------ | ---------- |
| **PPO**   | **~275**                 | 🥇 Fastest |
| **A2C**   | ~335                     | 🥈         |
| **DQN**   | ~450                     | 🥉         |
| REINFORCE | ~600                     | ❌ Slowest  |

---

# 🌍 **Generalization Performance**

| Metric               | PPO       | A2C   | DQN   |
| -------------------- | --------- | ----- | ----- |
| Novel positions      | 94%       | 87%   | 79%   |
| Unseen targets       | 89%       | 84%   | 71%   |
| Cross-phase transfer | 61%       | 58%   | 44%   |
| **Overall Score**    | **81.3%** | 76.3% | 64.7% |

### Key Takeaways

✔ PPO generalizes best
✔ DQN struggles on unseen targets
✔ All algorithms degrade in Phase 3 transfer

---

# 🏁 **Conclusion**

* **PPO is the optimal algorithm** for medical drone navigation in this project
* Strongest in **reward**, **stability**, **convergence**, and **generalization**
* **A2C is a strong runner-up**
* **DQN & REINFORCE are not ideal** for complex 3D navigation tasks

---

# 📦 **Project Structure**

```
📁 medical-drone-rl
│── 📄 README.md
│── 📄 Medical_drone.ipynb
│── 📁 models/
│── 📁 logs/
│── 📁 results/
│── 📄 requirements.txt
```

---

# ▶️ **How to Run**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Open Notebook

```bash
jupyter notebook Medical_drone.ipynb
```

### 3. Train a PPO model

```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
```

---

