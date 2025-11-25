import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import random
from typing import Dict, Tuple, Optional
import time

class MedicalDeliveryEnv(gym.Env):
    """Medical Delivery Environment with Curriculum Learning"""

    def __init__(self, render_mode=None, training_phase=1):
        super().__init__()
        
        self.training_phase = training_phase  # 1: Easy, 2: Medium, 3: Hard
        print("[INFO] Starting Training Phase {0}".format(self.training_phase))

        # Environment parameters
        self.city_size = 500
        self.max_steps = 1000
        self.current_step = 0

        # Action space for drone: [forward, right, up, yaw]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5, -0.8]),
            high=np.array([1.0, 1.0, 1.0, 0.8]),
            dtype=np.float32
        )

        # Enhanced Observation space
        obs_low = [-self.city_size, -self.city_size, 0, -10, -10, -5, -math.pi, -1, -1, -1, 0, 0]
        obs_high = [self.city_size, self.city_size, 100, 10, 10, 5, math.pi, 1, 1, 1, 1, 1]
        
        self.observation_space = spaces.Box(
            low=np.array(obs_low),
            high=np.array(obs_high),
            dtype=np.float32
        )

        # Key locations in Juba
        self.pharmacy_location = np.array([0, 0, 0])
        self.delivery_locations = {
            "konyo_konyo": np.array([80, 60, 0]),
            "munuki": np.array([-60, 75, 0]),
            "thongpiny": np.array([-90, -40, 0]),
            "juba_town": np.array([60, -75, 0])
        }

        # PyBullet setup
        self.physicsClient = None
        self.drone = None
        self.render_mode = render_mode
        self.obstacles = []
        self.markers = []

    def _setup_pybullet(self):
        """Initialize PyBullet simulation"""
        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Create ground
        self.ground = p.loadURDF("plane.urdf")

        # Create Juba city environment based on training phase
        self._create_juba_city()

        # Create drone
        self._create_drone()

        # Create markers
        self._create_markers()

        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=100,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 20]
        )

    def _create_juba_city(self):
        """Create simplified Juba city layout based on training phase"""
        
        # Phase 1: No obstacles for easy learning
        if self.training_phase == 1:
            print("[PHASE] Phase 1: No obstacles")
            return
            
        # Phase 2: Few obstacles
        elif self.training_phase == 2:
            print("[PHASE] Phase 2: Few obstacles")
            
            # Nile river only
            river_length = 120
            river_width = 20
            river_height = 0.5

            river_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[river_length/2, river_width/2, river_height/2])
            river_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[river_length/2, river_width/2, river_height/2],
                                             rgbaColor=[0.2, 0.4, 0.8, 0.7])

            river = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=river_collision,
                baseVisualShapeIndex=river_visual,
                basePosition=[0, 0, river_height/2]
            )
            self.obstacles.append(river)

            # Only 2 buildings
            building_positions = [
                [40, 60, 0], [-60, 70, 0]
            ]

        # Phase 3: Full environment
        else:
            print("[PHASE] Phase 3: Full environment")
            
            # Nile river
            river_length = 120
            river_width = 20
            river_height = 0.5

            river_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[river_length/2, river_width/2, river_height/2])
            river_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[river_length/2, river_width/2, river_height/2],
                                             rgbaColor=[0.2, 0.4, 0.8, 0.7])

            river = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=river_collision,
                baseVisualShapeIndex=river_visual,
                basePosition=[0, 0, river_height/2]
            )
            self.obstacles.append(river)

            # All buildings
            building_positions = [
                [40, 60, 0], [-60, 70, 0], [80, -30, 0], [-45, -80, 0]
            ]

        # Create buildings for phases 2 and 3
        if self.training_phase >= 2:
            for pos in building_positions:
                building_size = 15
                building_height = 8

                building_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[building_size/2, building_size/2, building_height/2])
                building_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[building_size/2, building_size/2, building_height/2],
                                                    rgbaColor=[0.6, 0.6, 0.6, 1])

                building = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=building_collision,
                    baseVisualShapeIndex=building_visual,
                    basePosition=[pos[0], pos[1], building_height/2]
                )
                self.obstacles.append(building)

    def _create_drone(self):
        """Create a realistic drone model"""
        # Drone body
        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.15])
        body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.15],
                                        rgbaColor=[0.1, 0.1, 0.8, 1])

        # Start higher in easier phases
        start_height = 15 if self.training_phase == 1 else 10 if self.training_phase == 2 else 5
        
        self.drone = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[self.pharmacy_location[0], self.pharmacy_location[1], start_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

        # Drone dynamics
        p.changeDynamics(self.drone, -1,
                        linearDamping=0.05,
                        angularDamping=0.1,
                        lateralFriction=0.1)

    def _create_markers(self):
        """Create markers for pharmacy and delivery points"""
        # Remove existing markers
        for marker in self.markers:
            try:
                p.removeBody(marker)
            except:
                pass
        self.markers = []

        # Pharmacy marker (Green cylinder)
        pharmacy_radius = 3
        pharmacy_length = 2

        pharmacy_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=pharmacy_radius, height=pharmacy_length)
        pharmacy_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=pharmacy_radius, length=pharmacy_length,
                                            rgbaColor=[0, 1, 0, 0.9])

        pharmacy_marker = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=pharmacy_collision,
            baseVisualShapeIndex=pharmacy_visual,
            basePosition=[self.pharmacy_location[0], self.pharmacy_location[1], pharmacy_length/2]
        )
        self.markers.append(pharmacy_marker)

        # Delivery point markers (Red cylinders)
        # In phase 1, only use the closest location
        if self.training_phase == 1:
            locations = {"konyo_konyo": self.delivery_locations["konyo_konyo"]}
        else:
            locations = self.delivery_locations

        for name, location in locations.items():
            delivery_radius = 2.5
            delivery_length = 1.5

            delivery_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=delivery_radius, height=delivery_length)
            delivery_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=delivery_radius, length=delivery_length,
                                                rgbaColor=[1, 0, 0, 0.9])

            delivery_marker = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=delivery_collision,
                baseVisualShapeIndex=delivery_visual,
                basePosition=[location[0], location[1], delivery_length/2]
            )
            self.markers.append(delivery_marker)

    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if self.physicsClient is not None:
            p.disconnect()

        self._setup_pybullet()

        # Reset drone position based on training phase
        start_height = 15 if self.training_phase == 1 else 10 if self.training_phase == 2 else 5
        
        p.resetBasePositionAndOrientation(
            self.drone,
            [self.pharmacy_location[0], self.pharmacy_location[1], start_height],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        # Reset state variables
        self.current_step = 0
        self.has_medicine = True
        
        # In phase 1, always use closest location
        if self.training_phase == 1:
            self.current_delivery_target = "konyo_konyo"
        else:
            self.current_delivery_target = random.choice(list(self.delivery_locations.keys()))
            
        self.delivery_completed = False
        self.return_to_pharmacy = False
        self.total_distance_traveled = 0.0
        self.last_position = np.array([self.pharmacy_location[0], self.pharmacy_location[1], start_height])
        self.successful_delivery = False
        self.successful_return = False
        self.previous_distance_to_target = float('inf')

        print("[MISSION] New mission: Deliver to {0}".format(self.current_delivery_target))

        return self._get_observation(), {}

    def _get_observation(self):
        """Get current observation with enhanced features"""
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        velocity, angular_velocity = p.getBaseVelocity(self.drone)

        # Convert quaternion to Euler angles
        euler = p.getEulerFromQuaternion(orientation)
        yaw = euler[2]

        # Get target position based on current mission phase
        if not self.delivery_completed:
            target_pos = self.delivery_locations[self.current_delivery_target]
            mission_phase = 0.0  # Going to delivery
        elif not self.return_to_pharmacy:
            target_pos = self.pharmacy_location
            mission_phase = 1.0  # Returning to pharmacy
        else:
            target_pos = self.pharmacy_location
            mission_phase = 2.0  # Mission complete

        # Relative position to target (3D)
        rel_pos = np.array(target_pos) - np.array(position)
        distance_to_target = np.linalg.norm(rel_pos[:2])  # Only horizontal distance

        # Normalize relative position
        if distance_to_target > 0:
            rel_pos_normalized = rel_pos / (distance_to_target + 1e-6)
        else:
            rel_pos_normalized = rel_pos

        # Battery level (decreases over time)
        battery_level = 1.0 - (self.current_step / self.max_steps)

        # Check if moving toward target (progress reward component)
        moving_toward_target = 1.0 if distance_to_target < self.previous_distance_to_target else 0.0
        self.previous_distance_to_target = distance_to_target

        observation = np.array([
            position[0], position[1], position[2],  # x, y, z position
            velocity[0], velocity[1], velocity[2],  # x, y, z velocity
            yaw,                                   # orientation (yaw)
            rel_pos_normalized[0],                 # normalized target x
            rel_pos_normalized[1],                 # normalized target y
            rel_pos[2] / 50.0,                     # height difference normalized
            mission_phase,                         # mission phase
            battery_level                          # battery level
        ], dtype=np.float32)

        return observation

    def step(self, action):
        """Execute one time step"""
        self.current_step += 1

        # Apply action to drone [forward, right, up, yaw]
        forward_force, right_force, up_force, yaw_torque = action

        # Smoother control
        max_force = 40
        max_torque = 8

        # Apply forces and torques
        p.applyExternalForce(self.drone, -1,
                           [forward_force * max_force, right_force * max_force, up_force * max_force],
                           [0, 0, 0], p.LINK_FRAME)

        p.applyExternalTorque(self.drone, -1,
                            [0, 0, yaw_torque * max_torque],
                            p.LINK_FRAME)

        # Step simulation
        p.stepSimulation()

        # Get new state
        observation = self._get_observation()
        position = p.getBasePositionAndOrientation(self.drone)[0]

        # Calculate enhanced reward
        reward = self._calculate_reward(position)

        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        # Update distance traveled
        self.total_distance_traveled += np.linalg.norm(np.array(position) - self.last_position)
        self.last_position = np.array(position)

        return observation, reward, terminated, truncated, {}

    def _calculate_reward(self, position):
        """ENHANCED REWARD STRUCTURE with progressive difficulty"""
        reward = 0
        pos_2d = np.array(position[:2])
        height = position[2]

        if not self.delivery_completed:
            # PHASE 1: Going to delivery location
            target_pos = self.delivery_locations[self.current_delivery_target]
            distance_to_target = np.linalg.norm(pos_2d - target_pos[:2])

            # LARGE POSITIVE REWARDS for progress (increased for easier learning)
            progress_reward = max(0, (1.0 - distance_to_target / 200.0)) * 5.0

            # Height management reward (optimal height based on phase)
            optimal_height = 20 if self.training_phase >= 2 else 25
            height_reward = -abs(height - optimal_height) * 0.005  # Softer penalty

            # Speed reward (moving toward target)
            velocity = p.getBaseVelocity(self.drone)[0]
            speed_toward_target = np.dot(velocity[:2], (target_pos[:2] - pos_2d)) / (distance_to_target + 1e-6)
            speed_reward = max(0, speed_toward_target) * 1.0  # Increased

            # Progress tracking reward
            progress_reward = 1.0 if distance_to_target < self.previous_distance_to_target else -0.1

            reward += progress_reward + height_reward + speed_reward

            # Check if delivery completed
            delivery_threshold = 10 if self.training_phase == 1 else 8  # Easier in phase 1
            if distance_to_target < delivery_threshold and 5 <= height <= (30 if self.training_phase == 1 else 25):
                reward += 200  # VERY LARGE POSITIVE REWARD
                self.delivery_completed = True
                self.has_medicine = False
                self.successful_delivery = True
                print("[SUCCESS] Medicine delivered to {0}! +200 reward".format(self.current_delivery_target))

        elif not self.return_to_pharmacy:
            # PHASE 2: Returning to pharmacy
            distance_to_pharmacy = np.linalg.norm(pos_2d - self.pharmacy_location[:2])

            # POSITIVE REWARDS for return progress
            progress_reward = max(0, (1.0 - distance_to_pharmacy / 200.0)) * 5.0

            # Height management reward
            optimal_height = 20 if self.training_phase >= 2 else 25
            height_reward = -abs(height - optimal_height) * 0.005

            # Speed reward
            velocity = p.getBaseVelocity(self.drone)[0]
            speed_toward_pharmacy = np.dot(velocity[:2], (self.pharmacy_location[:2] - pos_2d)) / (distance_to_pharmacy + 1e-6)
            speed_reward = max(0, speed_toward_pharmacy) * 1.0

            reward += progress_reward + height_reward + speed_reward

            # Check if returned to pharmacy
            return_threshold = 10 if self.training_phase == 1 else 8
            if distance_to_pharmacy < return_threshold and 5 <= height <= (30 if self.training_phase == 1 else 25):
                reward += 250  # VERY LARGE POSITIVE REWARD for mission completion
                self.return_to_pharmacy = True
                self.successful_return = True
                print("[MISSION_COMPLETE] Successfully returned to pharmacy! Mission complete! +250 reward")

        # VERY SMALL step penalty (reduced significantly)
        reward -= 0.001

        # Altitude limits (much softer penalties)
        if height < 3:
            reward -= 0.05
        if height > 40:
            reward -= 0.05

        # Check for collisions with buildings (only in phases 2-3)
        if self.training_phase >= 2:
            for obstacle in self.obstacles:
                closest_points = p.getClosestPoints(self.drone, obstacle, 2.0)  # Increased detection range
                if closest_points:
                    reward -= 2.0  # Reduced penalty
                    break

        # Large bonus for mission completion
        if self.successful_delivery and self.successful_return:
            reward += 100  # Completion bonus

        return reward

    def _check_termination(self):
        """Check if episode should terminate"""
        if self.return_to_pharmacy:
            return True

        # Check if drone is out of bounds (softer boundaries)
        position = p.getBasePositionAndOrientation(self.drone)[0]
        if (abs(position[0]) > self.city_size * 0.9 or
            abs(position[1]) > self.city_size * 0.9 or
            position[2] < -2 or position[2] > 60):
            return True

        return False

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            time.sleep(1/60)

    def close(self):
        """Close the environment"""
        if self.physicsClient is not None:
            p.disconnect()
            self.physicsClient = None
