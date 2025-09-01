"""
Multi-Objective Model Optimization Module
Implements adaptive quantization, structured pruning, and knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import copy

class AdaptiveMixedPrecisionQuantization:
    """Adaptive mixed-precision quantization with layer-wise bit allocation"""
    
    def __init__(self, model: nn.Module, 
                 bit_candidates: List[int] = [2, 4, 8, 16, 32],
                 memory_budget: float = 100.0):
        """
        Initialize quantization module
        
        Args:
            model: PyTorch model to quantize
            bit_candidates: Available bit widths
            memory_budget: Memory budget in MB
        """
        self.model = model
        self.bit_candidates = bit_candidates
        self.memory_budget = memory_budget
        self.layer_sensitivities = {}
        self.bit_allocation = {}
        
    def compute_layer_sensitivity(self, layer_name: str, 
                                 layer: nn.Module,
                                 data_loader: torch.utils.data.DataLoader) -> float:
        """
        Compute sensitivity score for a layer using Fisher Information
        
        Args:
            layer_name: Name of the layer
            layer: Layer module
            data_loader: Calibration data loader
        
        Returns:
            Sensitivity score
        """
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return 0.0
        
        # Compute gradient norm (simplified Fisher Information)
        grad_norm = 0.0
        weight = layer.weight.data
        
        # Estimate sensitivity using weight magnitude and layer FLOPs
        weight_norm = torch.norm(weight, p='fro').item()
        
        # Estimate FLOPs (simplified)
        if isinstance(layer, nn.Conv2d):
            flops = np.prod(weight.shape) * np.prod(layer.kernel_size)
        else:
            flops = np.prod(weight.shape)
        
        sensitivity = weight_norm * np.sqrt(flops)
        
        return sensitivity
    
    def optimize_bit_allocation(self, model: nn.Module,
                               calibration_loader: torch.utils.data.DataLoader) -> Dict[str, int]:
        """
        Optimize bit allocation across layers
        
        Args:
            model: Model to quantize
            calibration_loader: Data for calibration
        
        Returns:
            Bit allocation dictionary {layer_name: bits}
        """
        # Compute sensitivities for all layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                sensitivity = self.compute_layer_sensitivity(name, module, calibration_loader)
                self.layer_sensitivities[name] = sensitivity
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(self.layer_sensitivities.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Allocate bits based on sensitivity and budget
        total_params = sum(p.numel() for p in model.parameters())
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
        """
        Quantize a single layer to specified bit width
        
        Args:
            layer: Layer to quantize
            bits: Target bit width
        
        Returns:
            Quantized layer
        """
        if bits == 32:
            return layer  # No quantization needed
        
        # Simplified quantization (fake quantization for demonstration)
        # In practice, use torch.quantization or custom quantization
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
        """
        Apply optimized quantization to model
        
        Args:
            model: Model to quantize
        
        Returns:
            Quantized model
        """
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if name in self.bit_allocation:
                bits = self.bit_allocation[name]
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Replace with quantized version
                    parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                    child_name = name.split('.')[-1]
                    
                    quantized_layer = self.quantize_layer(module, bits)
                    
                    if parent_name:
                        parent = dict(quantized_model.named_modules())[parent_name]
                        setattr(parent, child_name, quantized_layer)
                    else:
                        setattr(quantized_model, child_name, quantized_layer)
        
        return quantized_model


class StructuredPruning:
    """Importance score-based structured pruning"""
    
    def __init__(self, model: nn.Module,
                 pruning_ratio: float = 0.5):
        """
        Initialize pruning module
        
        Args:
            model: Model to prune
            pruning_ratio: Target sparsity ratio
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_scores = {}
        
    def compute_taylor_importance(self, layer: nn.Module, 
                                 grad: torch.Tensor) -> torch.Tensor:
        """
        Compute Taylor expansion-based importance scores
        
        Args:
            layer: Layer module
            grad: Gradient tensor
        
        Returns:
            Channel importance scores
        """
        # Taylor importance: 0.5 * g^T * H * g (simplified as g^2)
        if isinstance(layer, nn.Conv2d):
            # Channel-wise importance for Conv2d
            importance = (grad ** 2).sum(dim=(0, 2, 3))
        elif isinstance(layer, nn.Linear):
            # Neuron-wise importance for Linear
            importance = (grad ** 2).sum(dim=0)
        else:
            importance = torch.zeros(1)
        
        return importance
    
    def compute_importance_scores(self, model: nn.Module,
                                 data_loader: torch.utils.data.DataLoader,
                                 criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for all layers
        
        Args:
            model: Model to analyze
            data_loader: Data for importance computation
            criterion: Loss function
        
        Returns:
            Importance scores dictionary
        """
        model.eval()
        importance_scores = {}
        
        # Register hooks to capture gradients
        handles = []
        gradients = {}
        
        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0]
            return hook
        
        # Register backward hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_backward_hook(hook_fn(name))
                handles.append(handle)
        
        # Compute gradients on calibration data
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 10:  # Use limited batches for efficiency
                break
            
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate importance scores
            for name, grad in gradients.items():
                module = dict(model.named_modules())[name]
                taylor_score = self.compute_taylor_importance(module, grad)
                
                if name not in importance_scores:
                    importance_scores[name] = taylor_score
                else:
                    importance_scores[name] += taylor_score
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Normalize scores
        for name in importance_scores:
            importance_scores[name] = importance_scores[name] / (batch_idx + 1)
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def prune_layer(self, layer: nn.Module, 
                   importance: torch.Tensor,
                   ratio: float) -> nn.Module:
        """
        Prune channels/neurons based on importance
        
        Args:
            layer: Layer to prune
            importance: Importance scores
            ratio: Pruning ratio
        
        Returns:
            Pruned layer
        """
        num_total = len(importance)
        num_prune = int(num_total * ratio)
        
        if num_prune == 0:
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
        """
        Apply structured pruning to model
        
        Args:
            model: Model to prune
        
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(model)
        
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
        """
        Initialize distillation module
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            initial_temperature: Initial distillation temperature
            final_temperature: Final distillation temperature
        """
        self.teacher = teacher_model
        self.student = student_model
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.current_epoch = 0
        
    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """
        Get temperature for current epoch
        
        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
        
        Returns:
            Temperature value
        """
        progress = epoch / total_epochs
        temp = self.initial_temp + (self.final_temp - self.initial_temp) * \
               np.exp(-progress)
        return temp
    
    def distillation_loss(self, student_output: torch.Tensor,
                         teacher_output: torch.Tensor,
                         target: torch.Tensor,
                         temperature: float,
                         alpha: float = 0.7) -> torch.Tensor:
        """
        Compute combined distillation loss
        
        Args:
            student_output: Student model output
            teacher_output: Teacher model output
            target: Ground truth labels
            temperature: Distillation temperature
            alpha: Weight for soft target loss
        
        Returns:
            Combined loss
        """
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
    
    def feature_matching_loss(self, student_features: List[torch.Tensor],
                            teacher_features: List[torch.Tensor],
                            beta: float = 0.2) -> torch.Tensor:
        """
        Compute feature matching loss
        
        Args:
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            beta: Weight for feature loss
        
        Returns:
            Feature matching loss
        """
        feature_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Normalize features
            s_feat_norm = F.normalize(s_feat.view(s_feat.size(0), -1), dim=1)
            t_feat_norm = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
            
            # MSE loss between normalized features
            feature_loss += F.mse_loss(s_feat_norm, t_feat_norm)
        
        return beta * feature_loss / len(student_features)
    
    def attention_transfer_loss(self, student_features: List[torch.Tensor],
                              teacher_features: List[torch.Tensor],
                              gamma: float = 0.1) -> torch.Tensor:
        """
        Compute attention transfer loss
        
        Args:
            student_features: Student feature maps
            teacher_features: Teacher feature maps
            gamma: Weight for attention loss
        
        Returns:
            Attention transfer loss
        """
        att_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Compute attention maps (sum of squared activations)
            s_att = (s_feat ** 2).mean(dim=1)
            t_att = (t_feat ** 2).mean(dim=1)
            
            # Normalize
            s_att = F.normalize(s_att.view(s_att.size(0), -1), dim=1)
            t_att = F.normalize(t_att.view(t_att.size(0), -1), dim=1)
            
            # MSE loss
            att_loss += F.mse_loss(s_att, t_att)
        
        return gamma * att_loss / len(student_features)
    
    def train_step(self, data: torch.Tensor, 
                  target: torch.Tensor,
                  epoch: int, 
                  total_epochs: int) -> Dict[str, float]:
        """
        Single training step with distillation
        
        Args:
            data: Input data
            target: Target labels
            epoch: Current epoch
            total_epochs: Total epochs
        
        Returns:
            Loss components dictionary
        """
        self.teacher.eval()
        self.student.train()
        
        # Get temperature
        temperature = self.get_temperature(epoch, total_epochs)
        
        # Forward pass
        with torch.no_grad():
            teacher_output = self.teacher(data)
        
        student_output = self.student(data)
        
        # Compute losses
        distill_loss = self.distillation_loss(
            student_output, teacher_output, target, temperature
        )
        
        losses = {
            'total': distill_loss.item(),
            'temperature': temperature
        }
        
        return losses


class MultiObjectiveOptimizer:
    """Combines all optimization techniques"""
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize multi-objective optimizer
        
        Args:
            model: Model to optimize
            config: Optimization configuration
        """
        self.model = model
        self.config = config
        
        # Initialize components
        self.quantizer = AdaptiveMixedPrecisionQuantization(
            model,
            bit_candidates=config.get('bit_candidates', [4, 8, 16, 32]),
            memory_budget=config.get('memory_budget', 100.0)
        )
        
        self.pruner = StructuredPruning(
            model,
            pruning_ratio=config.get('pruning_ratio', 0.5)
        )
        
        self.teacher_model = config.get('teacher_model', None)
        if self.teacher_model:
            self.distiller = AdaptiveKnowledgeDistillation(
                self.teacher_model,
                model,
                initial_temperature=config.get('initial_temp', 4.0),
                final_temperature=config.get('final_temp', 1.0)
            )
        
    def optimize(self, model: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                weights: np.ndarray) -> nn.Module:
        """
        Apply multi-objective optimization
        
        Args:
            model: Model to optimize
            data_loader: Calibration data
            weights: Optimization weights [latency, accuracy, memory]
        
        Returns:
            Optimized model
        """
        optimized_model = copy.deepcopy(model)
        
        # Weight-based optimization strategy
        latency_weight, accuracy_weight, memory_weight = weights
        
        # Apply quantization if memory is important
        if memory_weight > 0.3:
            print("Applying adaptive quantization...")
            self.quantizer.optimize_bit_allocation(optimized_model, data_loader)
            optimized_model = self.quantizer.apply_quantization(optimized_model)
        
        # Apply pruning if latency is important
        if latency_weight > 0.3:
            print("Applying structured pruning...")
            criterion = nn.CrossEntropyLoss()
            self.pruner.compute_importance_scores(optimized_model, data_loader, criterion)
            optimized_model = self.pruner.apply_pruning(optimized_model)
        
        # Note: Knowledge distillation would be applied during training
        # Here we just return the architecture-optimized model
        
        return optimized_model
    
    def get_model_stats(self, model: nn.Module) -> Dict[str, float]:
        """
        Get model statistics
        
        Args:
            model: Model to analyze
        
        Returns:
            Statistics dictionary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size (assuming float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Count layers by type
        conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        
        stats = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'conv_layers': conv_layers,
            'linear_layers': linear_layers
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)
            
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Initialize model and optimizer
    model = SimpleModel()
    
    config = {
        'bit_candidates': [4, 8, 16, 32],
        'memory_budget': 10.0,
        'pruning_ratio': 0.3,
        'teacher_model': None  # Would be a larger pre-trained model
    }
    
    optimizer = MultiObjectiveOptimizer(model, config)
    
    # Create dummy data loader
    dummy_data = torch.randn(32, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (32,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    
    # Optimize with specific weights
    weights = np.array([0.4, 0.4, 0.2])  # [latency, accuracy, memory]
    optimized_model = optimizer.optimize(model, data_loader, weights)
    
    # Get statistics
    original_stats = optimizer.get_model_stats(model)
    optimized_stats = optimizer.get_model_stats(optimized_model)
    
    print("Original Model Stats:")
    for key, value in original_stats.items():
        print(f"  {key}: {value}")
    
    print("\nOptimized Model Stats:")
    for key, value in optimized_stats.items():
        print(f"  {key}: {value}")
    
    compression_ratio = original_stats['model_size_mb'] / optimized_stats['model_size_mb']
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")
