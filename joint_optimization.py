"""
Joint Optimization Algorithm with Pareto Selection
Implements the complete TriOpt framework with multi-objective optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy
import time
from collections import defaultdict
import json

@dataclass
class OptimizationObjectives:
    """Optimization objectives"""
    latency: float      # ms
    accuracy: float     # percentage (0-100)
    memory: float       # MB
    energy: float       # mJ per inference

@dataclass
class Solution:
    """A solution in the optimization space"""
    model_config: Dict[str, Any]    # Model configuration
    system_config: Dict[str, Any]   # System configuration
    objectives: OptimizationObjectives
    constraints_satisfied: bool
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates another"""
        better_in_one = False
        
        obj1 = [self.objectives.latency, -self.objectives.accuracy, 
                self.objectives.memory, self.objectives.energy]
        obj2 = [other.objectives.latency, -other.objectives.accuracy,
                other.objectives.memory, other.objectives.energy]
        
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # Worse in this objective
                return False
            elif o1 < o2:  # Better in this objective
                better_in_one = True
        
        return better_in_one

class ParetoFrontier:
    """Maintains Pareto frontier of solutions"""
    
    def __init__(self):
        self.solutions = []
        
    def add(self, solution: Solution) -> bool:
        """
        Add solution to frontier if non-dominated
        
        Args:
            solution: Solution to add
            
        Returns:
            True if added, False if dominated
        """
        # Check if dominated by any existing solution
        for existing in self.solutions:
            if existing.dominates(solution):
                return False
        
        # Remove solutions dominated by new solution
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        
        # Add new solution
        self.solutions.append(solution)
        return True
    
    def get_best(self, weights: np.ndarray) -> Optional[Solution]:
        """
        Get best solution according to weighted objectives
        
        Args:
            weights: Weights for [latency, accuracy, memory, energy]
            
        Returns:
            Best solution or None
        """
        if not self.solutions:
            return None
        
        best_solution = None
        best_score = float('inf')
        
        for solution in self.solutions:
            # Compute weighted score (minimize)
            score = (
                weights[0] * solution.objectives.latency +
                weights[1] * (100 - solution.objectives.accuracy) +  # Maximize accuracy
                weights[2] * solution.objectives.memory +
                weights[3] * solution.objectives.energy
            )
            
            if score < best_score:
                best_score = score
                best_solution = solution
        
        return best_solution
    
    def size(self) -> int:
        """Get number of solutions in frontier"""
        return len(self.solutions)

class TriOptFramework:
    """
    Main TriOpt optimization framework
    Combines scene analysis, model optimization, and system adaptation
    """
    
    def __init__(self, model: nn.Module, 
                 hardware_platform: str = "auto",
                 config: Dict[str, Any] = None):
        """
        Initialize TriOpt framework
        
        Args:
            model: PyTorch model to optimize
            hardware_platform: Target hardware platform
            config: Configuration dictionary
        """
        self.original_model = copy.deepcopy(model)
        self.hardware_platform = self._detect_hardware(hardware_platform)
        self.config = config or self._get_default_config()
        
        # Initialize components (importing from previous modules)
        self.scene_analyzer = None  # Would import from scene_analyzer module
        self.model_optimizer = None  # Would import from model_optimization module
        self.system_adapter = None   # Would import from system_adaptation module
        
        # Optimization state
        self.pareto_frontier = ParetoFrontier()
        self.generation = 0
        self.convergence_history = []
        
    def _detect_hardware(self, platform: str) -> str:
        """Detect or validate hardware platform"""
        if platform == "auto":
            if torch.cuda.is_available():
                # Detect specific GPU
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "jetson" in gpu_name:
                    return "nvidia_jetson"
                elif "t4" in gpu_name or "v100" in gpu_name:
                    return "datacenter_gpu"
                else:
                    return "consumer_gpu"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "apple_silicon"
            else:
                import platform
                if "arm" in platform.machine().lower():
                    return "raspberry_pi"
                else:
                    return "x86_cpu"
        return platform
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on hardware"""
        base_config = {
            'population_size': 50,
            'num_generations': 100,
            'convergence_threshold': 0.01,
            'discretization_delta': 0.1,
            'bit_candidates': [4, 8, 16, 32],
            'pruning_ratios': np.arange(0, 0.9, 0.1).tolist(),
            'batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128],
        }
        
        # Hardware-specific adjustments
        hardware_configs = {
            'nvidia_jetson': {
                'preferred_precision': 'fp16',
                'use_tensorrt': True,
                'max_batch_size': 32,
                'optimization_focus': 'latency'
            },
            'raspberry_pi': {
                'preferred_precision': 'int8',
                'use_onnx': True,
                'max_batch_size': 8,
                'optimization_focus': 'memory'
            },
            'apple_silicon': {
                'preferred_precision': 'fp16',
                'use_coreml': True,
                'max_batch_size': 64,
                'optimization_focus': 'balanced'
            },
            'consumer_gpu': {
                'preferred_precision': 'fp16',
                'use_tensorrt': False,
                'max_batch_size': 128,
                'optimization_focus': 'throughput'
            }
        }
        
        if self.hardware_platform in hardware_configs:
            base_config.update(hardware_configs[self.hardware_platform])
        
        return base_config
    
    def evaluate_solution(self, model_config: Dict[str, Any],
                         system_config: Dict[str, Any],
                         data_loader: torch.utils.data.DataLoader) -> OptimizationObjectives:
        """
        Evaluate a solution's objectives
        
        Args:
            model_config: Model configuration
            system_config: System configuration
            data_loader: Evaluation data
            
        Returns:
            Optimization objectives
        """
        # Apply model configuration
        model = self._apply_model_config(self.original_model, model_config)
        
        # Apply system configuration
        model = self._apply_system_config(model, system_config)
        
        # Measure performance
        latency = self._measure_latency(model, data_loader, system_config)
        accuracy = self._measure_accuracy(model, data_loader)
        memory = self._measure_memory(model)
        energy = self._estimate_energy(model, latency)
        
        return OptimizationObjectives(
            latency=latency,
            accuracy=accuracy,
            memory=memory,
            energy=energy
        )
    
    def _apply_model_config(self, model: nn.Module, 
                           config: Dict[str, Any]) -> nn.Module:
        """Apply model optimization configuration"""
        optimized_model = copy.deepcopy(model)
        
        # Apply quantization
        if 'quantization_bits' in config:
            optimized_model = self._quantize_model(
                optimized_model, config['quantization_bits']
            )
        
        # Apply pruning
        if 'pruning_ratio' in config:
            optimized_model = self._prune_model(
                optimized_model, config['pruning_ratio']
            )
        
        return optimized_model
    
    def _apply_system_config(self, model: nn.Module,
                            config: Dict[str, Any]) -> nn.Module:
        """Apply system-level configuration"""
        # Apply precision
        if 'precision' in config:
            if config['precision'] == 'fp16' and torch.cuda.is_available():
                model = model.half()
            elif config['precision'] == 'int8':
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
        
        # Move to device
        device = config.get('device', 'cpu')
        model = model.to(device)
        
        return model
    
    def _quantize_model(self, model: nn.Module, bits: int) -> nn.Module:
        """Simplified quantization"""
        if bits == 32:
            return model
        
        # Fake quantization for demonstration
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    # Quantize weights
                    w = module.weight.data
                    scale = (w.max() - w.min()) / (2**bits - 1)
                    w_quant = torch.round(w / scale) * scale
                    module.weight.data = w_quant
        
        return model
    
    def _prune_model(self, model: nn.Module, ratio: float) -> nn.Module:
        """Simplified structured pruning"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune output channels
                num_channels = module.out_channels
                num_keep = int(num_channels * (1 - ratio))
                if num_keep > 0 and num_keep < num_channels:
                    # Get channel importance (simplified - using L2 norm)
                    importance = torch.norm(module.weight.data, dim=(1, 2, 3))
                    _, indices = torch.topk(importance, num_keep)
                    
                    # Create new layer with fewer channels
                    new_conv = nn.Conv2d(
                        module.in_channels, num_keep,
                        module.kernel_size, module.stride,
                        module.padding, module.dilation,
                        module.groups, module.bias is not None
                    )
                    new_conv.weight.data = module.weight.data[indices]
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data[indices]
                    
                    # Replace module (simplified - in practice need to handle connections)
                    module = new_conv
        
        return model
    
    def _measure_latency(self, model: nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        system_config: Dict[str, Any]) -> float:
        """Measure inference latency"""
        model.eval()
        device = system_config.get('device', 'cpu')
        batch_size = system_config.get('batch_size', 1)
        
        # Get sample input
        for data, _ in data_loader:
            data = data[:batch_size].to(device)
            break
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(data)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        num_iterations = 100
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(data)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        latency = ((end - start) / num_iterations) * 1000  # ms
        return latency
    
    def _measure_accuracy(self, model: nn.Module,
                         data_loader: torch.utils.data.DataLoader) -> float:
        """Measure model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                if total >= 1000:  # Limit evaluation samples
                    break
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
    
    def _measure_memory(self, model: nn.Module) -> float:
        """Measure model memory footprint"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate based on precision
        if next(model.parameters()).dtype == torch.float16:
            bytes_per_param = 2
        elif next(model.parameters()).dtype == torch.int8:
            bytes_per_param = 1
        else:
            bytes_per_param = 4
        
        memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
        return memory_mb
    
    def _estimate_energy(self, model: nn.Module, latency: float) -> float:
        """Estimate energy consumption"""
        # Simplified energy model
        base_power = {
            'nvidia_jetson': 15.0,      # Watts
            'raspberry_pi': 5.0,
            'apple_silicon': 20.0,
            'consumer_gpu': 100.0,
            'x86_cpu': 65.0
        }.get(self.hardware_platform, 50.0)
        
        # Energy = Power * Time
        energy_mj = base_power * (latency / 1000)  # Convert ms to seconds
        return energy_mj
    
    def generate_random_solution(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate random configuration"""
        model_config = {
            'quantization_bits': np.random.choice(self.config['bit_candidates']),
            'pruning_ratio': np.random.choice(self.config['pruning_ratios'])
        }
        
        system_config = {
            'batch_size': np.random.choice(self.config['batch_sizes']),
            'precision': np.random.choice(['fp32', 'fp16', 'int8']),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_threads': np.random.randint(1, 9)
        }
        
        return model_config, system_config
    
    def mutate_solution(self, model_config: Dict[str, Any],
                       system_config: Dict[str, Any],
                       mutation_rate: float = 0.2) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mutate a solution"""
        new_model_config = copy.deepcopy(model_config)
        new_system_config = copy.deepcopy(system_config)
        
        # Mutate model config
        if np.random.random() < mutation_rate:
            new_model_config['quantization_bits'] = np.random.choice(
                self.config['bit_candidates']
            )
        
        if np.random.random() < mutation_rate:
            new_model_config['pruning_ratio'] = np.random.choice(
                self.config['pruning_ratios']
            )
        
        # Mutate system config
        if np.random.random() < mutation_rate:
            new_system_config['batch_size'] = np.random.choice(
                self.config['batch_sizes']
            )
        
        if np.random.random() < mutation_rate:
            new_system_config['precision'] = np.random.choice(
                ['fp32', 'fp16', 'int8']
            )
        
        return new_model_config, new_system_config
    
    def optimize(self, data_loader: torch.utils.data.DataLoader,
                constraints: Dict[str, float],
                scene_weights: np.ndarray = None) -> Solution:
        """
        Main optimization loop
        
        Args:
            data_loader: Data for evaluation
            constraints: Performance constraints
            scene_weights: Scene-specific weights
            
        Returns:
            Best solution
        """
        if scene_weights is None:
            scene_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        print(f"Starting TriOpt optimization on {self.hardware_platform}")
        print(f"Constraints: {constraints}")
        print(f"Weights: {scene_weights}")
        
        population = []
        
        # Initialize population
        for _ in range(self.config['population_size']):
            model_cfg, system_cfg = self.generate_random_solution()
            population.append((model_cfg, system_cfg))
        
        # Evolution loop
        for gen in range(self.config['num_generations']):
            self.generation = gen
            
            # Evaluate population
            solutions = []
            for model_cfg, system_cfg in population:
                try:
                    objectives = self.evaluate_solution(
                        model_cfg, system_cfg, data_loader
                    )
                    
                    # Check constraints
                    satisfied = (
                        objectives.latency <= constraints.get('latency', float('inf')) and
                        objectives.accuracy >= constraints.get('accuracy', 0) and
                        objectives.memory <= constraints.get('memory', float('inf')) and
                        objectives.energy <= constraints.get('energy', float('inf'))
                    )
                    
                    solution = Solution(
                        model_config=model_cfg,
                        system_config=system_cfg,
                        objectives=objectives,
                        constraints_satisfied=satisfied
                    )
                    
                    solutions.append(solution)
                    
                    # Add to Pareto frontier
                    if satisfied:
                        self.pareto_frontier.add(solution)
                        
                except Exception as e:
                    print(f"Error evaluating solution: {e}")
                    continue
            
            # Select next generation
            if solutions:
                # Sort by weighted objective
                solutions.sort(
                    key=lambda s: (
                        scene_weights[0] * s.objectives.latency +
                        scene_weights[1] * (100 - s.objectives.accuracy) +
                        scene_weights[2] * s.objectives.memory +
                        scene_weights[3] * s.objectives.energy
                    )
                )
                
                # Keep best solutions
                num_keep = self.config['population_size'] // 2
                best_solutions = solutions[:num_keep]
                
                # Generate new population
                new_population = []
                for solution in best_solutions:
                    new_population.append(
                        (solution.model_config, solution.system_config)
                    )
                    
                    # Create mutated version
                    mutated = self.mutate_solution(
                        solution.model_config, solution.system_config
                    )
                    new_population.append(mutated)
                
                population = new_population[:self.config['population_size']]
            
            # Check convergence
            if gen > 10:
                recent_history = self.convergence_history[-10:]
                if len(recent_history) == 10:
                    variance = np.var([h['best_score'] for h in recent_history])
                    if variance < self.config['convergence_threshold']:
                        print(f"Converged at generation {gen}")
                        break
            
            # Log progress
            if gen % 10 == 0:
                best = self.pareto_frontier.get_best(scene_weights)
                if best:
                    print(f"Gen {gen}: Pareto size={self.pareto_frontier.size()}, "
                          f"Best: L={best.objectives.latency:.1f}ms, "
                          f"A={best.objectives.accuracy:.1f}%, "
                          f"M={best.objectives.memory:.1f}MB")
                    
                    self.convergence_history.append({
                        'generation': gen,
                        'pareto_size': self.pareto_frontier.size(),
                        'best_score': (
                            scene_weights[0] * best.objectives.latency +
                            scene_weights[1] * (100 - best.objectives.accuracy) +
                            scene_weights[2] * best.objectives.memory +
                            scene_weights[3] * best.objectives.energy
                        )
                    })
        
        # Return best solution
        best_solution = self.pareto_frontier.get_best(scene_weights)
        
        if best_solution:
            print(f"\nOptimization complete!")
            print(f"Best solution:")
            print(f"  Latency: {best_solution.objectives.latency:.2f} ms")
            print(f"  Accuracy: {best_solution.objectives.accuracy:.2f}%")
            print(f"  Memory: {best_solution.objectives.memory:.2f} MB")
            print(f"  Energy: {best_solution.objectives.energy:.2f} mJ")
            print(f"  Model config: {best_solution.model_config}")
            print(f"  System config: {best_solution.system_config}")
        
        return best_solution
    
    def export_solution(self, solution: Solution, 
                       output_path: str = "optimized_model.pt"):
        """
        Export optimized model
        
        Args:
            solution: Solution to export
            output_path: Output file path
        """
        # Apply configurations to get final model
        model = self._apply_model_config(self.original_model, solution.model_config)
        model = self._apply_system_config(model, solution.system_config)
        
        # Save model and configuration
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': solution.model_config,
            'system_config': solution.system_config,
            'objectives': {
                'latency': solution.objectives.latency,
                'accuracy': solution.objectives.accuracy,
                'memory': solution.objectives.memory,
                'energy': solution.objectives.energy
            },
            'hardware_platform': self.hardware_platform
        }, output_path)
        
        print(f"Model exported to {output_path}")
        
        # Export configuration as JSON
        config_path = output_path.replace('.pt', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': solution.model_config,
                'system_config': solution.system_config,
                'objectives': {
                    'latency': solution.objectives.latency,
                    'accuracy': solution.objectives.accuracy,
                    'memory': solution.objectives.memory,
                    'energy': solution.objectives.energy
                },
                'hardware_platform': self.hardware_platform
            }, f, indent=2)
        
        print(f"Configuration exported to {config_path}")


# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Initialize model and framework
    model = SimpleModel()
    triopt = TriOptFramework(model, hardware_platform="auto")
    
    # Create dummy dataset
    dummy_data = torch.randn(100, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    # Define constraints
    constraints = {
        'latency': 20.0,    # 20ms maximum
        'accuracy': 90.0,   # 90% minimum
        'memory': 50.0,     # 50MB maximum
        'energy': 100.0     # 100mJ maximum
    }
    
    # Scene-specific weights [latency, accuracy, memory, energy]
    # Higher weight = more important
    scene_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Prioritize latency
    
    # Run optimization
    best_solution = triopt.optimize(
        data_loader, 
        constraints,
        scene_weights
    )
    
    # Export solution
    if best_solution:
        triopt.export_solution(best_solution, "optimized_model.pt")
        
        # Display Pareto frontier
        print(f"\nPareto Frontier ({triopt.pareto_frontier.size()} solutions):")
        for i, sol in enumerate(triopt.pareto_frontier.solutions[:5]):
            print(f"  Solution {i+1}: L={sol.objectives.latency:.1f}ms, "
                  f"A={sol.objectives.accuracy:.1f}%, "
                  f"M={sol.objectives.memory:.1f}MB, "
                  f"E={sol.objectives.energy:.1f}mJ")
