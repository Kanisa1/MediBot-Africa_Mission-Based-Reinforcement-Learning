
````markdown
# MediBot Africa - Mission-Based Reinforcement Learning

## Overview
MediBot Africa is an **AI-powered healthcare simulation system** designed to support malaria defense operations in South Sudan. This system trains intelligent agents using **Reinforcement Learning (RL)** to manage outbreaks efficiently by allocating resources like diagnostic kits, medicines, alerts, and consultations.

The project demonstrates how AI can enhance decision-making in **public health interventions**, providing a scalable digital tool for healthcare crisis management.

---

## Key Features
- **Custom Malaria Defense Environment** simulating villages, clinics, pharmacies, and outbreaks.
- **Trained RL Agents** using:
  - **DQN** (Deep Q-Network)
  - **PPO** (Proximal Policy Optimization)
  - **A2C** (Advantage Actor-Critic)
- **Baseline Random Agent** for performance comparison.
- **Visualization**:
  - Interactive grid displaying agent, outbreaks, and resources.
  - Live stats panel showing step, cumulative reward, lives saved, outbreaks contained, and resources used.
  - Legend for quick reference.
- **Video Recording** of agent gameplay for presentations (`videos/` folder).

---

## Installation
1. Clone the repository:
```bash
git clone <repo-url>
cd MediBot-Africa_Mission-Based-Reinforcement-Learning
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Train agents** (DQN, PPO, A2C):

```bash
python dqn_training.py
python pg_training.py
```

2. **Run demonstration**:

```bash
python main.py
```

* The script runs **random agent** as baseline, followed by DQN, PPO, and A2C agents.
* Average rewards and final performance comparison are printed.
* Agent gameplay videos are saved automatically in the `videos/` folder.

---

## Mission Goal

The system aims to **maximize lives saved and contain malaria outbreaks** by training AI agents to optimize resource allocation under limited supplies, demonstrating the **impact of RL in healthcare decision-making**.

---

## Presentation Ready

* GIF videos of each agent are saved in `videos/` and can be shown during presentations.
* Statistics provide clear quantitative performance comparison between agents and baseline.

---

## License

MIT License

```

---