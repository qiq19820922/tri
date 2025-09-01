"""
Scene Analyzer Module with Learnable Weights
Implements hierarchical scene analysis and policy network for adaptive optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ApplicationCategory(Enum):
    """Application deployment categories"""
    REAL_TIME_VISION = "real_time_vision"
    MEDICAL_IMAGING = "medical_imaging"
    VIDEO_ANALYSIS = "video_analysis"
    NLP = "natural_language_processing"
    AUDIO = "audio_processing"

@dataclass
class PerformanceRequirements:
    """Performance requirements for deployment"""
    latency_target: float  # ms
    memory_budget: float   # MB
    power_budget: float    # Watts
    accuracy_req: float    # minimum accuracy %
    batch_size: int
    input_resolution: Tuple[int, int, int]  # (H, W, C)

@dataclass
class EnvironmentalContext:
    """Environmental deployment context"""
    data_modality: str
    network_conditions: str  # "stable", "unstable", "offline"
    usage_pattern: str  # "continuous", "burst", "periodic"
    temporal_variance: float  # variance in workload

class SceneEncoder(nn.Module):
    """Encodes scene representation from hierarchical components"""
    
    def __init__(self, scene_dim: int = 64):
        super().__init__()
        self.scene_dim = scene_dim
        
        # Category embedding
        self.category_embed = nn.Embedding(len(ApplicationCategory), 16)
        
        # Performance feature extractor
        self.perf_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 24)
        )
        
        # Environmental feature extractor
        self.env_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 24)
        )
        
        # Scene fusion
        self.scene_fusion = nn.Sequential(
            nn.Linear(16 + 24 + 24, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, scene_dim)
        )
    
    def forward(self, category: torch.Tensor, 
                perf_reqs: torch.Tensor, 
                env_context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode scene
        
        Args:
            category: Application category index [batch_size]
            perf_reqs: Performance requirements [batch_size, 6]
            env_context: Environmental context [batch_size, 4]
        
        Returns:
            Scene encoding [batch_size, scene_dim]
        """
        cat_embed = self.category_embed(category)
        perf_feat = self.perf_encoder(perf_reqs)
        env_feat = self.env_encoder(env_context)
        
        combined = torch.cat([cat_embed, perf_feat, env_feat], dim=-1)
        scene = self.scene_fusion(combined)
        
        return scene

class PolicyNetwork(nn.Module):
    """Policy network for learning scene-specific optimization weights"""
    
    def __init__(self, scene_dim: int = 64, num_objectives: int = 3):
        super().__init__()
        
        # Feature extractor φ(s)
        self.feature_extractor = nn.Sequential(
            nn.Linear(scene_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Policy head π_θ
        self.policy_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_objectives)
        )
        
        self.num_objectives = num_objectives
    
    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        Generate optimization weights for given scene
        
        Args:
            scene: Scene encoding [batch_size, scene_dim]
        
        Returns:
            Weights λ for objectives [batch_size, num_objectives]
        """
        features = self.feature_extractor(scene)
        logits = self.policy_head(features)
        weights = torch.softmax(logits, dim=-1)
        
        return weights

class SceneAwareOptimizer:
    """Scene-aware weight learning with policy gradient"""
    
    def __init__(self, scene_dim: int = 64, 
                 num_objectives: int = 3,
                 learning_rate: float = 1e-3):
        
        self.scene_encoder = SceneEncoder(scene_dim)
        self.policy_network = PolicyNetwork(scene_dim, num_objectives)
        
        # Combine parameters for joint training
        params = list(self.scene_encoder.parameters()) + \
                 list(self.policy_network.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # History for policy gradient
        self.reward_history = []
        self.log_prob_history = []
        
    def encode_scene(self, category: ApplicationCategory,
                     perf_reqs: PerformanceRequirements,
                     env_context: EnvironmentalContext) -> torch.Tensor:
        """Encode deployment scene into vector representation"""
        
        # Convert to tensors
        cat_idx = list(ApplicationCategory).index(category)
        cat_tensor = torch.tensor([cat_idx], dtype=torch.long)
        
        perf_tensor = torch.tensor([[
            perf_reqs.latency_target,
            perf_reqs.memory_budget,
            perf_reqs.power_budget,
            perf_reqs.accuracy_req,
            perf_reqs.batch_size,
            np.prod(perf_reqs.input_resolution)
        ]], dtype=torch.float32)
        
        # Encode environmental context (simplified)
        env_tensor = torch.tensor([[
            hash(env_context.data_modality) % 100 / 100.0,
            hash(env_context.network_conditions) % 100 / 100.0,
            hash(env_context.usage_pattern) % 100 / 100.0,
            env_context.temporal_variance
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            scene = self.scene_encoder(cat_tensor, perf_tensor, env_tensor)
        
        return scene
    
    def get_optimization_weights(self, scene: torch.Tensor,
                                 training: bool = False) -> np.ndarray:
        """
        Get optimization weights for given scene
        
        Args:
            scene: Scene encoding
            training: Whether in training mode (record gradients)
        
        Returns:
            Weights λ for [latency, accuracy, memory] objectives
        """
        if training:
            weights = self.policy_network(scene)
            # Sample action from distribution for exploration
            dist = torch.distributions.Categorical(weights)
            action = dist.sample()
            self.log_prob_history.append(dist.log_prob(action))
        else:
            with torch.no_grad():
                weights = self.policy_network(scene)
        
        return weights.cpu().numpy().squeeze()
    
    def update_policy(self, reward: float):
        """
        Update policy network using REINFORCE algorithm
        
        Args:
            reward: Optimization performance reward
        """
        self.reward_history.append(reward)
        
        # Update every batch
        if len(self.reward_history) >= 8:
            self._policy_gradient_update()
    
    def _policy_gradient_update(self):
        """Perform policy gradient update"""
        
        # Normalize rewards
        rewards = torch.tensor(self.reward_history, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, reward in zip(self.log_prob_history, rewards):
            policy_loss.append(-log_prob * reward)
        
        if policy_loss:
            loss = torch.stack(policy_loss).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), 1.0
            )
            self.optimizer.step()
        
        # Clear history
        self.reward_history.clear()
        self.log_prob_history.clear()
    
    def compute_reward(self, latency: float, accuracy: float, 
                      memory: float, targets: Dict[str, float]) -> float:
        """
        Compute reward based on optimization results
        
        Args:
            latency: Achieved latency (ms)
            accuracy: Achieved accuracy (%)
            memory: Model size (MB)
            targets: Target requirements
        
        Returns:
            Scalar reward value
        """
        # Reward components
        latency_reward = max(0, 1 - latency / targets['latency'])
        accuracy_reward = min(1, accuracy / targets['accuracy'])
        memory_reward = max(0, 1 - memory / targets['memory'])
        
        # Weighted combination
        reward = 0.4 * latency_reward + 0.4 * accuracy_reward + 0.2 * memory_reward
        
        # Penalty for constraint violations
        if latency > targets['latency']:
            reward -= 0.2
        if accuracy < targets['accuracy']:
            reward -= 0.3
        if memory > targets['memory']:
            reward -= 0.1
        
        return reward

# Example usage
if __name__ == "__main__":
    # Initialize scene-aware optimizer
    optimizer = SceneAwareOptimizer()
    
    # Define deployment scenario
    category = ApplicationCategory.REAL_TIME_VISION
    perf_reqs = PerformanceRequirements(
        latency_target=10.0,  # 10ms
        memory_budget=100.0,   # 100MB
        power_budget=5.0,      # 5W
        accuracy_req=0.95,     # 95%
        batch_size=1,
        input_resolution=(224, 224, 3)
    )
    env_context = EnvironmentalContext(
        data_modality="image",
        network_conditions="stable",
        usage_pattern="continuous",
        temporal_variance=0.1
    )
    
    # Encode scene
    scene = optimizer.encode_scene(category, perf_reqs, env_context)
    
    # Get optimization weights
    weights = optimizer.get_optimization_weights(scene, training=True)
    print(f"Optimization weights: {weights}")
    print(f"  Latency weight: {weights[0]:.3f}")
    print(f"  Accuracy weight: {weights[1]:.3f}")
    print(f"  Memory weight: {weights[2]:.3f}")
    
    # Simulate optimization result and update policy
    achieved_latency = 8.5
    achieved_accuracy = 0.94
    achieved_memory = 85.0
    
    targets = {
        'latency': perf_reqs.latency_target,
        'accuracy': perf_reqs.accuracy_req,
        'memory': perf_reqs.memory_budget
    }
    
    reward = optimizer.compute_reward(
        achieved_latency, achieved_accuracy, achieved_memory, targets
    )
    optimizer.update_policy(reward)
    
    print(f"\nReward: {reward:.3f}")
