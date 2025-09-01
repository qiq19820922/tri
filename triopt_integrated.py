"""
TriOpt: Joint Optimization Framework for Edge AI Deployment
Integrated module combining all optimization components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque, OrderedDict
from enum import Enum
import copy
import time
import json
import os
import platform
import psutil
import threading
import queue
import multiprocessing as mp

# ============================================================================
# Data Classes and Enums
# ============================================================================

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

@dataclass
class HardwareSpecs:
    """Hardware specifications"""
    platform_name: str
    device_type: str  # cpu, gpu, tpu, npu
    compute_capability: float
    memory_gb: float
    cores: int
    frequency_mhz: float
    supported_precisions: List[str]
    has_tensorrt: bool
    has_coreml: bool
    has_onnx: bool
    power_budget_watts: float

# ============================================================================
# Scene Analysis Module
# ============================================================================

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
        """Forward pass to encode scene"""
        cat_embed = self.category_embed(category)
        perf_feat = self.perf_encoder(perf_reqs)
        env_feat = self.env_encoder(env_context)
        
        combined = torch.cat([cat_embed, perf_feat, env_feat], dim=-1)
        scene = self.scene_fusion(combined)
        
        return scene

class PolicyNetwork(nn.Module):
    """Policy network for learning scene-specific optimization weights"""
    
    def __init__(self, scene_dim: int = 64, num_objectives: int = 4):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(scene_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_objectives)
        )
        
        self.num_objectives = num_objectives
    
    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """Generate optimization weights for given scene"""
        features = self.feature_extractor(scene)
        logits = self.policy_head(features)
        weights = torch.softmax(logits, dim=-1)
        
        return weights

# ============================================================================
# Hardware Detection Module
# ============================================================================

class HardwareDetector:
    """Detects and profiles hardware capabilities"""
    
    def __init__(self):
        self.specs = self._detect_hardware()
        
    def _detect_hardware(self) -> HardwareSpecs:
        """Comprehensive hardware detection"""
        
        # Detect device type and capabilities
        if torch.cuda.is_available():
            return self._detect_nvidia_gpu()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return self._get_apple_silicon_specs()
        elif 'arm' in platform.machine().lower():
            return self._detect_arm_device()
        else:
            return self._detect_x86_cpu()
    
    def _detect_nvidia_gpu(self) -> HardwareSpecs:
        """Detect NVIDIA GPU specifications"""
        
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Determine platform
        if 'jetson' in device_name.lower():
            platform_name = "jetson"
            power = 15
        elif any(x in device_name.lower() for x in ['t4', 'v100', 'a100']):
            platform_name = "datacenter_gpu"
            power = 250
        else:
            platform_name = "consumer_gpu"
            power = 150
        
        # Supported precisions
        supported_precisions = ['fp32']
        if capability[0] >= 6:
            supported_precisions.append('fp16')
        if capability[0] >= 7:
            supported_precisions.extend(['int8', 'tf32'])
        if capability[0] >= 8:
            supported_precisions.append('bf16')
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='gpu',
            compute_capability=capability[0] + capability[1] * 0.1,
            memory_gb=memory_gb,
            cores=torch.cuda.get_device_properties(0).multi_processor_count,
            frequency_mhz=torch.cuda.get_device_properties(0).clock_rate / 1000,
            supported_precisions=supported_precisions,
            has_tensorrt=self._check_tensorrt(),
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=power
        )
    
    def _detect_arm_device(self) -> HardwareSpecs:
        """Detect ARM device (Raspberry Pi, etc.)"""
        
        platform_name = 'raspberry_pi'
        cores = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='cpu',
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=cores,
            frequency_mhz=1500,
            supported_precisions=['fp32', 'int8'],
            has_tensorrt=False,
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=7
        )
    
    def _detect_x86_cpu(self) -> HardwareSpecs:
        """Detect x86 CPU specifications"""
        
        cpu_info = platform.processor()
        cores = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        platform_name = 'intel_x86' if 'intel' in cpu_info.lower() else 'amd_x86'
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='cpu',
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=cores,
            frequency_mhz=2000,
            supported_precisions=['fp32', 'int8', 'fp16'],
            has_tensorrt=False,
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=65
        )
    
    def _get_apple_silicon_specs(self) -> HardwareSpecs:
        """Get Apple Silicon specifications"""
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return HardwareSpecs(
            platform_name='apple_silicon',
            device_type='npu',
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=8,
            frequency_mhz=3200,
            supported_precisions=['fp32', 'fp16', 'int8'],
            has_tensorrt=False,
            has_coreml=True,
            has_onnx=True,
            power_budget_watts=20
        )
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

# ============================================================================
# Model Optimization Module
# ============================================================================

class AdaptiveMixedPrecisionQuantization:
    """Adaptive mixed-precision quantization with layer-wise bit allocation"""
    
    def __init__(self, model: nn.Module, 
                 bit_candidates: List[int] = [2, 4, 8, 16, 32],
                 memory_budget: float = 100.0):
        self.model = model
        self.bit_candidates = bit_candidates
        self.memory_budget = memory_budget
        self.layer_sensitivities = {}
        self.bit_allocation = {}
        
    def compute_layer_sensitivity(self, layer_name: str, 
                                 layer: nn.Module) -> float:
        """Compute sensitivity score for a layer using Fisher Information"""
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return 0.0
        
        weight = layer.weight.data
        weight_norm = torch.norm(weight, p='fro').item()
        
        if isinstance(layer, nn.Conv2d):
            flops = np.prod(weight.shape) * np.prod(layer.kernel_size)
        else:
            flops = np.prod(weight.shape)
        
        sensitivity = weight_norm * np.sqrt(flops)
        
        return sensitivity
    
    def optimize_bit_allocation(self, model: nn.Module) -> Dict[str, int]:
        """Optimize bit allocation across layers"""
        # Compute sensitivities for all layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                sensitivity = self.compute_layer_sensitivity(name, module)
                self.layer_sensitivities[name] = sensitivity
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(self.layer_sensitivities.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Allocate bits based on sensitivity and budget
        allocated_memory = 0.0
        
        for layer_name, sensitivity in sorted_layers:
            # Try bit widths from high to low
            for bits in sorted(self.bit_candidates, reverse=True):
                layer = dict(model.named_modules())[layer_name]
                params = sum(p.numel() for p in layer.parameters())
                
                # Calculate memory with current bit width
                layer_memory = (params * bits) / (8 * 1024 * 1024)  # MB
                
                if allocated_memory + layer_memory <= self.memory_budget:
                    self.bit_allocation[layer_name] = bits
                    allocated_memory += layer_memory
                    break
            else:
                # Use minimum bits if budget exceeded
                self.bit_allocation[layer_name] = min(self.bit_candidates)
        
        return self.bit_allocation
    
    def quantize_layer(self, layer: nn.Module, bits: int) -> nn.Module:
        """Quantize a single layer to specified bit width"""
        if bits == 32:
            return layer  # No quantization needed
        
        # Simplified quantization (fake quantization for demonstration)
        def quantize_tensor(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
            qmin = -(2 ** (num_bits - 1))
            qmax = 2 ** (num_bits - 1) - 1
            
            scale = (tensor.max() - tensor.min()) / (qmax - qmin)
            zero_point = qmin - tensor.min() / scale
            
            # Quantize and dequantize
            q_tensor = torch.round(tensor / scale + zero_point)
            q_tensor = torch.clamp(q_tensor, qmin, qmax)
            dq_tensor = (q_tensor - zero_point) * scale
            
            return dq_tensor
        
        # Quantize weights
        if hasattr(layer, 'weight'):
            layer.weight.data = quantize_tensor(layer.weight.data, bits)
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data = quantize_tensor(layer.bias.data, bits)
        
        return layer
    
    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply optimized quantization to model"""
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if name in self.bit_allocation:
                bits = self.bit_allocation[name]
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    self.quantize_layer(module, bits)
        
        return quantized_model

class StructuredPruning:
    """Importance score-based structured pruning"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_scores = {}
        
    def compute_importance_scores(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute importance scores for all layers"""
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    # Channel-wise importance for Conv2d
                    importance = torch.norm(module.weight.data, dim=(1, 2, 3))
                elif isinstance(module, nn.Linear):
                    # Neuron-wise importance for Linear
                    importance = torch.norm(module.weight.data, dim=1)
                
                self.importance_scores[name] = importance
        
        return self.importance_scores
    
    def prune_layer(self, layer: nn.Module, 
                   importance: torch.Tensor,
                   ratio: float) -> nn.Module:
        """Prune channels/neurons based on importance"""
        num_total = len(importance)
        num_prune = int(num_total * ratio)
        
        if num_prune == 0 or num_prune >= num_total:
            return layer
        
        # Get indices to keep
        _, indices = torch.sort(importance, descending=True)
        keep_indices = indices[:num_total - num_prune]
        
        # Create pruned layer
        if isinstance(layer, nn.Conv2d):
            # Prune output channels
            new_layer = nn.Conv2d(
                layer.in_channels,
                len(keep_indices),
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias=layer.bias is not None
            )
            
            # Copy weights
            new_layer.weight.data = layer.weight.data[keep_indices]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_indices]
                
        elif isinstance(layer, nn.Linear):
            # Prune output neurons
            new_layer = nn.Linear(
                layer.in_features,
                len(keep_indices),
                bias=layer.bias is not None
            )
            
            # Copy weights
            new_layer.weight.data = layer.weight.data[keep_indices]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_indices]
        else:
            new_layer = layer
        
        return new_layer
    
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model"""
        pruned_model = copy.deepcopy(model)
        
        self.compute_importance_scores(pruned_model)
        
        for name, module in list(pruned_model.named_modules()):
            if name in self.importance_scores:
                importance = self.importance_scores[name]
                pruned_layer = self.prune_layer(module, importance, self.pruning_ratio)
                
                # Replace layer in model
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(pruned_model.named_modules())[parent_name]
                    setattr(parent, child_name, pruned_layer)
                else:
                    setattr(pruned_model, child_name, pruned_layer)
        
        return pruned_model

class AdaptiveKnowledgeDistillation:
    """Adaptive knowledge distillation with temperature scheduling"""
    
    def __init__(self, teacher_model: nn.Module,
                 student_model: nn.Module,
                 initial_temperature: float = 4.0,
                 final_temperature: float = 1.0):
        self.teacher = teacher_model
        self.student = student_model
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.current_epoch = 0
        
    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Get temperature for current epoch"""
        progress = epoch / total_epochs
        temp = self.initial_temp + (self.final_temp - self.initial_temp) * \
               np.exp(-progress)
        return temp
    
    def distillation_loss(self, student_output: torch.Tensor,
                         teacher_output: torch.Tensor,
                         target: torch.Tensor,
                         temperature: float,
                         alpha: float = 0.7) -> torch.Tensor:
        """Compute combined distillation loss"""
        # Soft targets loss
        soft_loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_output, target)
        
        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return loss

# ============================================================================
# System Adaptation Module
# ============================================================================

class DynamicBatchOptimizer:
    """Dynamic batch size optimization with online learning"""
    
    def __init__(self, initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 256,
                 learning_rate: float = 0.01):
        self.current_batch_size = initial_batch_size
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.learning_rate = learning_rate
        self.history = deque(maxlen=100)
        self.update_counter = 0
        
    def measure_performance(self, model: nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int,
                           device: str = 'cpu') -> Tuple[float, float]:
        """Measure throughput and latency for given batch size"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure latency
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        num_iterations = 10
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency = (total_time / num_iterations) * 1000  # ms
        throughput = (batch_size * num_iterations) / total_time  # images/sec
        
        return throughput, latency
    
    def update_batch_size(self, model: nn.Module,
                         input_shape: Tuple[int, ...],
                         device: str = 'cpu') -> int:
        """Update batch size based on online learning"""
        self.update_counter += 1
        
        # Only update every N iterations
        if self.update_counter % 100 != 0:
            return self.current_batch_size
        
        # Measure current performance
        curr_throughput, curr_latency = self.measure_performance(
            model, input_shape, self.current_batch_size, device
        )
        curr_efficiency = curr_throughput / (curr_latency + 1e-6)
        
        # Try adjacent batch sizes
        candidates = [(self.current_batch_size, curr_efficiency)]
        
        # Try smaller batch size
        if self.current_batch_size > self.min_batch:
            smaller_batch = max(self.min_batch, int(self.current_batch_size * 0.8))
            s_throughput, s_latency = self.measure_performance(
                model, input_shape, smaller_batch, device
            )
            s_efficiency = s_throughput / (s_latency + 1e-6)
            candidates.append((smaller_batch, s_efficiency))
        
        # Try larger batch size
        if self.current_batch_size < self.max_batch:
            larger_batch = min(self.max_batch, int(self.current_batch_size * 1.2))
            
            try:
                l_throughput, l_latency = self.measure_performance(
                    model, input_shape, larger_batch, device
                )
                l_efficiency = l_throughput / (l_latency + 1e-6)
                candidates.append((larger_batch, l_efficiency))
            except RuntimeError as e:  # Out of memory
                if "out of memory" in str(e).lower():
                    self.max_batch = self.current_batch_size
        
        # Select best batch size
        best_batch, best_efficiency = max(candidates, key=lambda x: x[1])
        self.current_batch_size = best_batch
        
        # Store in history
        self.history.append({
            'batch_size': self.current_batch_size,
            'throughput': curr_throughput,
            'latency': curr_latency,
            'efficiency': curr_efficiency
        })
        
        return self.current_batch_size

class PrecisionModeSelector:
    """Hardware-aware precision mode selection"""
    
    def __init__(self, supported_precisions: List[str] = None):
        self.device = self._detect_device()
        self.supported_precisions = supported_precisions or self._detect_supported_precisions()
        
        # Precision performance profiles (relative speedup)
        self.precision_profiles = {
            'fp32': {'speedup': 1.0, 'accuracy_loss': 0.0},
            'fp16': {'speedup': 1.8, 'accuracy_loss': 0.001},
            'int8': {'speedup': 3.2, 'accuracy_loss': 0.01},
            'int4': {'speedup': 5.5, 'accuracy_loss': 0.03}
        }
        
        self.current_precision = 'fp32'
        
    def _detect_device(self) -> str:
        """Detect available compute device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _detect_supported_precisions(self) -> List[str]:
        """Detect supported precision modes based on hardware"""
        supported = ['fp32']  # Always supported
        
        if self.device == 'cuda':
            # Check CUDA capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:  # Volta or newer
                    supported.extend(['fp16', 'int8'])
                if capability[0] >= 8:  # Ampere or newer
                    supported.append('int4')
        
        elif self.device == 'mps':
            # Apple Silicon supports fp16
            supported.append('fp16')
        
        # CPU always supports int8
        if 'int8' not in supported:
            supported.append('int8')
        
        return supported
    
    def select_precision(self, accuracy_requirement: float,
                        latency_target: float) -> str:
        """Select optimal precision mode"""
        best_precision = 'fp32'
        best_score = 0.0
        
        for precision in self.supported_precisions:
            profile = self.precision_profiles.get(precision, {})
            
            # Check if accuracy loss is acceptable
            accuracy_loss = profile.get('accuracy_loss', 0)
            if accuracy_loss > (1 - accuracy_requirement):
                continue
            
            # Compute score: maximize speedup while meeting constraints
            speedup = profile.get('speedup', 1.0)
            score = speedup / (accuracy_loss + 1e-6)
            
            if score > best_score:
                best_score = score
                best_precision = precision
        
        self.current_precision = best_precision
        return best_precision
    
    def apply_precision(self, model: nn.Module, 
                       precision: str) -> nn.Module:
        """Apply precision mode to model"""
        if precision == 'fp32':
            return model.float()
        
        elif precision == 'fp16':
            if self.device in ['cuda', 'mps']:
                return model.half()
            else:
                print(f"FP16 not supported on {self.device}, using FP32")
                return model.float()
        
        elif precision == 'int8':
            # Simplified INT8 quantization
            model_int8 = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            return model_int8
        
        else:
            print(f"Unknown precision: {precision}")
            return model

# ============================================================================
# Pareto Frontier Module
# ============================================================================

class ParetoFrontier:
    """Maintains Pareto frontier of solutions"""
    
    def __init__(self):
        self.solutions = []
        
    def add(self, solution: Solution) -> bool:
        """Add solution to frontier if non-dominated"""
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
        """Get best solution according to weighted objectives"""
        if not self.solutions:
            return None
        
        best_solution = None
        best_score = float('inf')
        
        for solution in self.solutions:
            # Compute weighted score (minimize)
            score = (
                weights[0] * solution.objectives.latency +
                weights[1] * (100 - solution.objectives.accuracy) +
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

# ============================================================================
# Main TriOpt Framework
# ============================================================================

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
        
        # Initialize hardware detector
        self.hardware_detector = HardwareDetector()
        self.hardware_specs = self.hardware_detector.specs
        self.hardware_platform = self.hardware_specs.platform_name
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.scene_encoder = SceneEncoder()
        self.policy_network = PolicyNetwork()
        
        # Model optimization components
        self.quantizer = AdaptiveMixedPrecisionQuantization(
            model,
            bit_candidates=self.config.get('bit_candidates', [4, 8, 16, 32]),
            memory_budget=self.config.get('memory_budget', 100.0)
        )
        self.pruner = StructuredPruning(
            model,
            pruning_ratio=self.config.get('pruning_ratio', 0.3)
        )
        
        # System adaptation components
        self.batch_optimizer = DynamicBatchOptimizer()
        self.precision_selector = PrecisionModeSelector(
            supported_precisions=self.hardware_specs.supported_precisions
        )
        
        # Optimization state
        self.pareto_frontier = ParetoFrontier()
        self.generation = 0
        self.convergence_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on hardware"""
        base_config = {
            'population_size': 50,
            'num_generations': 100,
            'convergence_threshold': 0.01,
            'bit_candidates': [4, 8, 16, 32],
            'pruning_ratios': np.arange(0, 0.9, 0.1).tolist(),
            'batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128],
            'memory_budget': 100.0,
            'teacher_model': None
        }
        
        return base_config
    
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
        
        # Encode environmental context
        env_tensor = torch.tensor([[
            hash(env_context.data_modality) % 100 / 100.0,
            hash(env_context.network_conditions) % 100 / 100.0,
            hash(env_context.usage_pattern) % 100 / 100.0,
            env_context.temporal_variance
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            scene = self.scene_encoder(cat_tensor, perf_tensor, env_tensor)
        
        return scene
    
    def get_optimization_weights(self, scene: torch.Tensor) -> np.ndarray:
        """Get optimization weights for given scene"""
        with torch.no_grad():
            weights = self.policy_network(scene)
        
        return weights.cpu().numpy().squeeze()
    
    def apply_model_optimizations(self, model: nn.Module,
                                 model_config: Dict[str, Any]) -> nn.Module:
        """Apply model optimization configuration"""
        optimized_model = copy.deepcopy(model)
        
        # Apply quantization
        if 'quantization_bits' in model_config:
            self.quantizer.optimize_bit_allocation(optimized_model)
            optimized_model = self.quantizer.apply_quantization(optimized_model)
        
        # Apply pruning
        if 'pruning_ratio' in model_config:
            self.pruner.pruning_ratio = model_config['pruning_ratio']
            optimized_model = self.pruner.apply_pruning(optimized_model)
        
        return optimized_model
    
    def apply_system_optimizations(self, model: nn.Module,
                                  system_config: Dict[str, Any]) -> nn.Module:
        """Apply system-level configuration"""
        # Apply precision
        if 'precision' in system_config:
            model = self.precision_selector.apply_precision(
                model, system_config['precision']
            )
        
        # Move to device
        device = system_config.get('device', 'cpu')
        model = model.to(device)
        
        return model
    
    def evaluate_solution(self, model: nn.Module,
                         model_config: Dict[str, Any],
                         system_config: Dict[str, Any],
                         data_loader: torch.utils.data.DataLoader) -> OptimizationObjectives:
        """Evaluate a solution's objectives"""
        # Apply optimizations
        optimized_model = self.apply_model_optimizations(model, model_config)
        optimized_model = self.apply_system_optimizations(optimized_model, system_config)
        
        # Measure performance
        device = system_config.get('device', 'cpu')
        batch_size = system_config.get('batch_size', 1)
        
        # Get sample input
        for data, target in data_loader:
            data = data[:batch_size].to(device)
            target = target[:batch_size].to(device)
            break
        
        # Measure latency
        optimized_model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = optimized_model(data)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        num_iterations = 100
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = optimized_model(data)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        latency = ((end - start) / num_iterations) * 1000  # ms
        
        # Measure accuracy (simplified)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                if total >= 100:  # Limit evaluation samples
                    break
                
                data = data.to(device)
                target = target.to(device)
                output = optimized_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # Measure memory
        total_params = sum(p.numel() for p in optimized_model.parameters())
        if next(optimized_model.parameters()).dtype == torch.float16:
            bytes_per_param = 2
        elif next(optimized_model.parameters()).dtype == torch.int8:
            bytes_per_param = 1
        else:
            bytes_per_param = 4
        
        memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
        
        # Estimate energy
        base_power = self.hardware_specs.power_budget_watts
        energy_mj = base_power * (latency / 1000)
        
        return OptimizationObjectives(
            latency=latency,
            accuracy=accuracy,
            memory=memory_mb,
            energy=energy_mj
        )
    
    def generate_random_solution(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate random configuration"""
        model_config = {
            'quantization_bits': np.random.choice(self.config['bit_candidates']),
            'pruning_ratio': np.random.choice(self.config['pruning_ratios'])
        }
        
        system_config = {
            'batch_size': np.random.choice(self.config['batch_sizes']),
            'precision': np.random.choice(self.hardware_specs.supported_precisions),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_threads': np.random.randint(1, 9)
        }
        
        return model_config, system_config
    
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
        print(f"Hardware specs: {self.hardware_specs.device_type}, "
              f"{self.hardware_specs.memory_gb:.1f}GB, "
              f"{self.hardware_specs.cores} cores")
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
                        self.original_model, model_cfg, system_cfg, data_loader
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
                    mutated_model_cfg = copy.deepcopy(solution.model_config)
                    mutated_system_cfg = copy.deepcopy(solution.system_config)
                    
                    # Mutate with 20% probability
                    if np.random.random() < 0.2:
                        mutated_model_cfg['quantization_bits'] = np.random.choice(
                            self.config['bit_candidates']
                        )
                    if np.random.random() < 0.2:
                        mutated_model_cfg['pruning_ratio'] = np.random.choice(
                            self.config['pruning_ratios']
                        )
                    if np.random.random() < 0.2:
                        mutated_system_cfg['batch_size'] = np.random.choice(
                            self.config['batch_sizes']
                        )
                    
                    new_population.append((mutated_model_cfg, mutated_system_cfg))
                
                population = new_population[:self.config['population_size']]
            
            # Log progress
            if gen % 10 == 0:
                best = self.pareto_frontier.get_best(scene_weights)
                if best:
                    print(f"Gen {gen}: Pareto size={self.pareto_frontier.size()}, "
                          f"Best: L={best.objectives.latency:.1f}ms, "
                          f"A={best.objectives.accuracy:.1f}%, "
                          f"M={best.objectives.memory:.1f}MB")
        
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
        """Export optimized model"""
        # Apply configurations to get final model
        model = self.apply_model_optimizations(self.original_model, solution.model_config)
        model = self.apply_system_optimizations(model, solution.system_config)
        
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
