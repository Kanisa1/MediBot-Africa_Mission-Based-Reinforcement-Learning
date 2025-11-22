import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from enum import Enum

class CellType(Enum):
    EMPTY = 0
    VILLAGE = 1
    CLINIC = 2
    PHARMACY = 3
    OUTBREAK = 4
    AGENT = 5

class MalariaDefenseEnv(gym.Env):
    """
    Custom Gymnasium environment for Malaria Defense Agent in South Sudan
    """
    
    def __init__(self, grid_size=10, max_steps=200):
        super(MalariaDefenseEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 8 directions + 4 special actions
        self.action_space = spaces.Discrete(12)
        
        # Observation space: grid + agent status + resources
        self.observation_space = spaces.Box(
            low=0, high=10, 
            shape=(grid_size, grid_size, 6), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.resources = {
            'diagnostic_kits': 10,
            'medicines': 15,
            'alerts_sent': 0,
            'consultations': 0
        }
        self.villages = []
        self.clinics = []
        self.pharmacies = []
        self.outbreaks = []
        
        self._generate_environment()
        
        self.lives_saved = 0
        self.outbreaks_contained = 0
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _generate_environment(self):
        num_villages = random.randint(8, 12)
        for _ in range(num_villages):
            pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if pos != self.agent_pos:
                self.villages.append({
                    'pos': pos,
                    'population': random.randint(50, 200),
                    'malaria_risk': random.uniform(0.1, 0.8),
                    'has_outbreak': False
                })
                self.grid[pos[0], pos[1]] = CellType.VILLAGE.value
        
        num_clinics = random.randint(2, 4)
        for _ in range(num_clinics):
            pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if self.grid[pos[0], pos[1]] == 0:
                self.clinics.append({
                    'pos': pos,
                    'capacity': random.randint(20, 50),
                    'supplies': random.randint(5, 15)
                })
                self.grid[pos[0], pos[1]] = CellType.CLINIC.value
        
        num_pharmacies = random.randint(3, 5)
        for _ in range(num_pharmacies):
            pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if self.grid[pos[0], pos[1]] == 0:
                self.pharmacies.append({
                    'pos': pos,
                    'medicine_stock': random.randint(10, 30)
                })
                self.grid[pos[0], pos[1]] = CellType.PHARMACY.value
        
        self._generate_outbreaks()
    
    def _generate_outbreaks(self):
        for village in self.villages:
            if random.random() < village['malaria_risk'] * 0.3:
                village['has_outbreak'] = True
                self.outbreaks.append({
                    'pos': village['pos'],
                    'severity': random.uniform(0.3, 1.0),
                    'duration': random.randint(5, 15)
                })
                pos = village['pos']
                self.grid[pos[0], pos[1]] = CellType.OUTBREAK.value
    
    def step(self, action):
        self.current_step += 1
        reward = 0
        
        if action < 8:
            reward += self._move_agent(action)
        elif action == 8:
            reward += self._deploy_diagnostic()
        elif action == 9:
            reward += self._distribute_medicine()
        elif action == 10:
            reward += self._send_alert()
        elif action == 11:
            reward += self._consult_doctor()
        
        reward += self._update_environment()
        
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        reward += self.lives_saved * 10
        reward += self.outbreaks_contained * 15
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _move_agent(self, action):
        directions = [
            [-1, 0], [-1, 1], [0, 1], [1, 1],
            [1, 0], [1, -1], [0, -1], [-1, -1]
        ]
        new_pos = [self.agent_pos[0] + directions[action][0],
                   self.agent_pos[1] + directions[action][1]]
        
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.agent_pos = new_pos
            return 1
        return -5
    
    def _deploy_diagnostic(self):
        if self.resources['diagnostic_kits'] <= 0:
            return -10
        for village in self.villages:
            if self._distance(self.agent_pos, village['pos']) <= 1:
                self.resources['diagnostic_kits'] -= 1
                return 20 if village['has_outbreak'] else 10
        return -5
    
    def _distribute_medicine(self):
        if self.resources['medicines'] <= 0:
            return -10
        for outbreak in self.outbreaks[:]:
            if self._distance(self.agent_pos, outbreak['pos']) <= 1:
                self.resources['medicines'] -= 1
                outbreak['severity'] *= 0.7
                self.lives_saved += int(outbreak['severity'] * 10)
                if outbreak['severity'] < 0.2:
                    self.outbreaks.remove(outbreak)
                    self.outbreaks_contained += 1
                    return 50
                return 25
        return -5
    
    def _send_alert(self):
        self.resources['alerts_sent'] += 1
        reward = 0
        for village in self.villages:
            if self._distance(self.agent_pos, village['pos']) <= 2:
                village['malaria_risk'] *= 0.9
                reward += 5
        return max(reward, -2)
    
    def _consult_doctor(self):
        self.resources['consultations'] += 1
        for clinic in self.clinics:
            if self._distance(self.agent_pos, clinic['pos']) <= 1:
                return 15
        return 5
    
    def _update_environment(self):
        reward = 0
        new_outbreaks = []
        for village in self.villages:
            if not village['has_outbreak'] and random.random() < village['malaria_risk'] * 0.05:
                village['has_outbreak'] = True
                new_outbreaks.append({
                    'pos': village['pos'],
                    'severity': random.uniform(0.2, 0.6),
                    'duration': random.randint(5, 15)
                })
                reward -= 20
        self.outbreaks.extend(new_outbreaks)
        for outbreak in self.outbreaks[:]:
            outbreak['duration'] -= 1
            if outbreak['duration'] <= 0:
                self.outbreaks.remove(outbreak)
                reward -= 30
        return reward
    
    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _check_termination(self):
        if len(self.outbreaks) > 8:
            return True
        if self.resources['diagnostic_kits'] <= 0 and self.resources['medicines'] <= 0 and len(self.outbreaks) > 0:
            return True
        return False
    
    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 6))
        obs[:, :, 0] = self.grid
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1
        for outbreak in self.outbreaks:
            pos = outbreak['pos']
            obs[pos[0], pos[1], 2] = outbreak['severity']
        for village in self.villages:
            pos = village['pos']
            obs[pos[0], pos[1], 3] = village['malaria_risk']
        obs[:, :, 4] = self.resources['diagnostic_kits'] / 20.0
        obs[:, :, 5] = self.resources['medicines'] / 30.0
        return obs.astype(np.float32)
    
    def _get_info(self):
        return {
            'lives_saved': self.lives_saved,
            'outbreaks_contained': self.outbreaks_contained,
            'active_outbreaks': len(self.outbreaks),
            'resources': self.resources.copy(),
            'step': self.current_step
        }
    
    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Malaria Defense Environment")
        
        self.screen.fill((255, 255, 255))
        cell_size = 600 // self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                if [i, j] == self.agent_pos:
                    color = (0, 0, 255)
                elif self.grid[i, j] == CellType.VILLAGE.value:
                    color = (0, 255, 0)
                elif self.grid[i, j] == CellType.CLINIC.value:
                    color = (255, 255, 0)
                elif self.grid[i, j] == CellType.PHARMACY.value:
                    color = (255, 0, 255)
                elif self.grid[i, j] == CellType.OUTBREAK.value:
                    color = (255, 0, 0)
                else:
                    color = (240, 240, 240)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        pygame.display.flip()
    
    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
