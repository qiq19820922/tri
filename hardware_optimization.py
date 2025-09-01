"""
Hardware Detection and Platform-Specific Optimization Module
Implements automatic hardware detection and platform-specific optimization strategies
"""

import torch
import torch.nn as nn
import numpy as np
import platform
import subprocess
import psutil
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
    
class HardwareDetector:
    """Detects and profiles hardware capabilities"""
    
    def __init__(self):
        self.specs = self._detect_hardware()
        
    def _detect_hardware(self) -> HardwareSpecs:
        """Comprehensive hardware detection"""
        
        # Basic system info
        system_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        # Detect device type and capabilities
        if torch.cuda.is_available():
            return self._detect_nvidia_gpu()
        elif self._detect_coral_tpu():
            return self._detect_edge_tpu()
        elif self._detect_apple_silicon():
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
        
        # Detect specific platforms
        if 'jetson' in device_name.lower():
            platform_name = self._detect_jetson_model()
        elif 't4' in device_name.lower():
            platform_name = "tesla_t4"
        elif 'v100' in device_name.lower():
            platform_name = "tesla_v100"
        elif 'a100' in device_name.lower():
            platform_name = "tesla_a100"
        elif any(x in device_name.lower() for x in ['rtx', 'gtx']):
            platform_name = "consumer_gpu"
        else:
            platform_name = "generic_nvidia_gpu"
        
        # Determine supported precisions based on compute capability
        supported_precisions = ['fp32']
        if capability[0] >= 6:  # Pascal or newer
            supported_precisions.append('fp16')
        if capability[0] >= 7:  # Volta or newer
            supported_precisions.extend(['int8', 'tf32'])
        if capability[0] >= 8:  # Ampere or newer
            supported_precisions.append('bf16')
        
        # Check for TensorRT
        has_tensorrt = self._check_tensorrt()
        
        # Power budget (approximate)
        power_budgets = {
            'jetson_nano': 10,
            'jetson_xavier_nx': 15,
            'jetson_orin': 30,
            'tesla_t4': 70,
            'tesla_v100': 250,
            'tesla_a100': 400,
            'consumer_gpu': 150
        }
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='gpu',
            compute_capability=capability[0] + capability[1] * 0.1,
            memory_gb=memory_gb,
            cores=torch.cuda.get_device_properties(0).multi_processor_count,
            frequency_mhz=torch.cuda.get_device_properties(0).clock_rate / 1000,
            supported_precisions=supported_precisions,
            has_tensorrt=has_tensorrt,
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=power_budgets.get(platform_name, 100)
        )
    
    def _detect_jetson_model(self) -> str:
        """Detect specific Jetson model"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'nano' in model.lower():
                    return 'jetson_nano'
                elif 'xavier' in model.lower():
                    return 'jetson_xavier_nx'
                elif 'orin' in model.lower():
                    return 'jetson_orin'
        except:
            pass
        return 'jetson_unknown'
    
    def _detect_coral_tpu(self) -> bool:
        """Check if Google Coral Edge TPU is available"""
        try:
            # Check for Edge TPU runtime
            result = subprocess.run(['edgetpu_compiler', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _detect_edge_tpu(self) -> HardwareSpecs:
        """Get Edge TPU specifications"""
        return HardwareSpecs(
            platform_name='google_coral_tpu',
            device_type='tpu',
            compute_capability=1.0,  # Edge TPU version 1
            memory_gb=0.5,  # 512MB typical
            cores=1,  # Single TPU core
            frequency_mhz=500,
            supported_precisions=['int8'],  # Edge TPU only supports INT8
            has_tensorrt=False,
            has_coreml=False,
            has_onnx=False,  # Uses TFLite
            power_budget_watts=2.0
        )
    
    def _detect_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        return (platform.system() == 'Darwin' and 
                platform.machine() == 'arm64')
    
    def _get_apple_silicon_specs(self) -> HardwareSpecs:
        """Get Apple Silicon specifications"""
        
        # Detect specific chip
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            chip_name = result.stdout.strip()
            
            if 'm1' in chip_name.lower():
                platform_name = 'apple_m1'
                cores = 8
                power = 20
            elif 'm2' in chip_name.lower():
                platform_name = 'apple_m2'
                cores = 10
                power = 25
            elif 'm3' in chip_name.lower():
                platform_name = 'apple_m3'
                cores = 12
                power = 30
            else:
                platform_name = 'apple_silicon'
                cores = 8
                power = 20
        except:
            platform_name = 'apple_silicon'
            cores = 8
            power = 20
        
        # Get memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='npu',  # Neural Engine
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=cores,
            frequency_mhz=3200,  # Approximate
            supported_precisions=['fp32', 'fp16', 'int8'],
            has_tensorrt=False,
            has_coreml=True,
            has_onnx=True,
            power_budget_watts=power
        )
    
    def _detect_arm_device(self) -> HardwareSpecs:
        """Detect ARM device (Raspberry Pi, etc.)"""
        
        # Try to detect Raspberry Pi model
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'raspberry pi 4' in model.lower():
                    platform_name = 'raspberry_pi_4'
                    cores = 4
                    freq = 1500
                    power = 7
                elif 'raspberry pi 5' in model.lower():
                    platform_name = 'raspberry_pi_5'
                    cores = 4
                    freq = 2400
                    power = 12
                else:
                    platform_name = 'generic_arm'
                    cores = psutil.cpu_count()
                    freq = 1000
                    power = 5
        except:
            platform_name = 'generic_arm'
            cores = psutil.cpu_count()
            freq = 1000
            power = 5
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='cpu',
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=cores,
            frequency_mhz=freq,
            supported_precisions=['fp32', 'int8'],
            has_tensorrt=False,
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=power
        )
    
    def _detect_x86_cpu(self) -> HardwareSpecs:
        """Detect x86 CPU specifications"""
        
        cpu_info = platform.processor()
        cores = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect CPU brand
        if 'intel' in cpu_info.lower():
            platform_name = 'intel_x86'
            power = 65
        elif 'amd' in cpu_info.lower():
            platform_name = 'amd_x86'
            power = 65
        else:
            platform_name = 'generic_x86'
            power = 50
        
        # Check for AVX support (for better SIMD operations)
        supported_precisions = ['fp32', 'int8']
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            if 'avx2' in info.get('flags', []):
                supported_precisions.append('fp16')
        except:
            pass
        
        return HardwareSpecs(
            platform_name=platform_name,
            device_type='cpu',
            compute_capability=1.0,
            memory_gb=memory_gb,
            cores=cores,
            frequency_mhz=2000,  # Approximate
            supported_precisions=supported_precisions,
            has_tensorrt=False,
            has_coreml=False,
            has_onnx=True,
            power_budget_watts=power
        )
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False


class PlatformOptimizer(ABC):
    """Abstract base class for platform-specific optimizers"""
    
    @abstractmethod
    def optimize_model(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply platform-specific optimizations"""
        pass
    
    @abstractmethod
    def get_optimization_config(self, constraints: Dict) -> Dict:
        """Get platform-specific optimization configuration"""
        pass


class JetsonOptimizer(PlatformOptimizer):
    """NVIDIA Jetson-specific optimizations"""
    
    def optimize_model(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply Jetson-specific optimizations"""
        
        # Enable mixed precision
        model = model.half()
        
        # Fuse BatchNorm layers
        model = self._fuse_bn(model)
        
        # Apply TensorRT if available
        if config.get('use_tensorrt', False):
            model = self._apply_tensorrt(model, config)
        
        return model
    
    def _fuse_bn(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN layers for better performance"""
        
        # Simplified fusion (in practice, use torch.fx or TensorRT)
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                fused = []
                i = 0
                while i < len(module):
                    if i < len(module) - 1:
                        if isinstance(module[i], nn.Conv2d) and \
                           isinstance(module[i+1], nn.BatchNorm2d):
                            # Fuse Conv and BN
                            fused_conv = self._fuse_conv_bn(module[i], module[i+1])
                            fused.append(fused_conv)
                            i += 2
                            continue
                    fused.append(module[i])
                    i += 1
                
                if len(fused) < len(module):
                    setattr(model, name, nn.Sequential(*fused))
        
        return model
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fuse a Conv2d-BatchNorm2d pair"""
        
        # Get parameters
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
        
        # BN parameters
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_eps = bn.eps
        bn_weight = bn.weight
        bn_bias = bn.bias
        
        # Compute fused parameters
        bn_std = torch.sqrt(bn_var + bn_eps)
        
        # Scale conv weights
        w_fused = w * (bn_weight / bn_std).view(-1, 1, 1, 1)
        
        # Compute fused bias
        b_fused = (b - bn_mean) * bn_weight / bn_std + bn_bias
        
        # Create fused conv
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels,
            conv.kernel_size, conv.stride,
            conv.padding, conv.dilation,
            conv.groups, bias=True
        )
        
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        
        return fused_conv
    
    def _apply_tensorrt(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply TensorRT optimization (placeholder)"""
        # In practice, convert to TensorRT engine
        print("TensorRT optimization would be applied here")
        return model
    
    def get_optimization_config(self, constraints: Dict) -> Dict:
        """Get Jetson-specific configuration"""
        return {
            'precision': 'fp16',
            'use_tensorrt': True,
            'batch_size': min(32, constraints.get('max_batch', 32)),
            'quantization_bits': 16,
            'pruning_ratio': 0.3,
            'optimization_level': 3
        }


class CoralTPUOptimizer(PlatformOptimizer):
    """Google Coral Edge TPU optimizations"""
    
    def optimize_model(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply Edge TPU optimizations"""
        
        # Edge TPU requires INT8 quantization
        model = self._quantize_for_tpu(model)
        
        # Ensure compatible operations
        model = self._ensure_tpu_compatibility(model)
        
        return model
    
    def _quantize_for_tpu(self, model: nn.Module) -> nn.Module:
        """Quantize model for Edge TPU (INT8 only)"""
        
        # Apply INT8 quantization
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        return model
    
    def _ensure_tpu_compatibility(self, model: nn.Module) -> nn.Module:
        """Ensure all operations are TPU-compatible"""
        
        # Replace incompatible layers
        for name, module in model.named_children():
            # TPU doesn't support certain operations
            if isinstance(module, nn.AdaptiveAvgPool2d):
                # Replace with fixed-size pooling
                setattr(model, name, nn.AvgPool2d(7))
        
        return model
    
    def get_optimization_config(self, constraints: Dict) -> Dict:
        """Get TPU-specific configuration"""
        return {
            'precision': 'int8',
            'quantization_bits': 8,
            'batch_size': 1,  # Edge TPU typically processes single images
            'pruning_ratio': 0.5,  # Aggressive pruning for edge
            'use_tflite': True
        }


class RaspberryPiOptimizer(PlatformOptimizer):
    """Raspberry Pi-specific optimizations"""
    
    def optimize_model(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply Raspberry Pi optimizations"""
        
        # Aggressive pruning for limited compute
        model = self._aggressive_prune(model, config.get('pruning_ratio', 0.6))
        
        # INT8 quantization for efficiency
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        return model
    
    def _aggressive_prune(self, model: nn.Module, ratio: float) -> nn.Module:
        """Apply aggressive structured pruning"""
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune channels with lowest L2 norm
                weight = module.weight.data
                num_channels = weight.shape[0]
                num_keep = int(num_channels * (1 - ratio))
                
                if num_keep > 0:
                    importance = torch.norm(weight, dim=(1, 2, 3))
                    threshold = torch.topk(importance, num_keep, largest=False)[0].max()
                    mask = importance > threshold
                    
                    # Apply mask
                    module.weight.data = weight[mask]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[mask]
        
        return model
    
    def get_optimization_config(self, constraints: Dict) -> Dict:
        """Get Raspberry Pi configuration"""
        return {
            'precision': 'int8',
            'quantization_bits': 8,
            'batch_size': 1,
            'pruning_ratio': 0.6,
            'num_threads': 4,
            'use_onnx': True
        }


class AppleSiliconOptimizer(PlatformOptimizer):
    """Apple Silicon (M1/M2/M3) optimizations"""
    
    def optimize_model(self, model: nn.Module, config: Dict) -> nn.Module:
        """Apply Apple Silicon optimizations"""
        
        # Use FP16 for Neural Engine
        model = model.half()
        
        # Optimize for CoreML if needed
        if config.get('use_coreml', False):
            model = self._prepare_for_coreml(model)
        
        return model
    
    def _prepare_for_coreml(self, model: nn.Module) -> nn.Module:
        """Prepare model for CoreML conversion"""
        
        # Ensure compatible operations
        for name, module in model.named_children():
            # Replace operations not supported by CoreML
            if isinstance(module, nn.GroupNorm):
                # Replace with BatchNorm
                setattr(model, name, nn.BatchNorm2d(module.num_channels))
        
        return model
    
    def get_optimization_config(self, constraints: Dict) -> Dict:
        """Get Apple Silicon configuration"""
        return {
            'precision': 'fp16',
            'quantization_bits': 16,
            'batch_size': 64,
            'pruning_ratio': 0.2,
            'use_coreml': True,
            'use_metal': True
        }


class PlatformSpecificOptimizer:
    """Main platform-specific optimization controller"""
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.specs = self.detector.specs
        self.optimizer = self._get_platform_optimizer()
        
    def _get_platform_optimizer(self) -> PlatformOptimizer:
        """Get appropriate platform optimizer"""
        
        platform_map = {
            'jetson': JetsonOptimizer,
            'coral': CoralTPUOptimizer,
            'raspberry': RaspberryPiOptimizer,
            'apple': AppleSiliconOptimizer
        }
        
        # Select optimizer based on platform
        platform_name = self.specs.platform_name.lower()
        
        for key, optimizer_class in platform_map.items():
            if key in platform_name:
                return optimizer_class()
        
        # Default optimizer
        return JetsonOptimizer()  # Generic GPU optimizer
    
    def optimize(self, model: nn.Module, 
                constraints: Dict[str, float]) -> Tuple[nn.Module, Dict]:
        """
        Apply platform-specific optimizations
        
        Args:
            model: Model to optimize
            constraints: Performance constraints
            
        Returns:
            Optimized model and configuration
        """
        
        print(f"Detected platform: {self.specs.platform_name}")
        print(f"Device type: {self.specs.device_type}")
        print(f"Memory: {self.specs.memory_gb:.1f} GB")
        print(f"Supported precisions: {self.specs.supported_precisions}")
        
        # Get platform-specific configuration
        config = self.optimizer.get_optimization_config(constraints)
        
        # Adjust based on hardware capabilities
        if config['precision'] not in self.specs.supported_precisions:
            # Fallback to best available precision
            if 'fp16' in self.specs.supported_precisions:
                config['precision'] = 'fp16'
            elif 'int8' in self.specs.supported_precisions:
                config['precision'] = 'int8'
            else:
                config['precision'] = 'fp32'
        
        # Apply optimizations
        optimized_model = self.optimizer.optimize_model(model, config)
        
        # Add hardware info to config
        config['hardware_specs'] = {
            'platform': self.specs.platform_name,
            'device_type': self.specs.device_type,
            'memory_gb': self.specs.memory_gb,
            'power_budget': self.specs.power_budget_watts
        }
        
        return optimized_model, config
    
    def export_deployment_package(self, model: nn.Module, 
                                 config: Dict,
                                 output_dir: str = "deployment"):
        """
        Export complete deployment package
        
        Args:
            model: Optimized model
            config: Optimization configuration
            output_dir: Output directory
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model in appropriate format
        if config.get('use_tensorrt'):
            # Export for TensorRT
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input, 
                            f"{output_dir}/model.onnx",
                            opset_version=11)
            print(f"Model exported to {output_dir}/model.onnx for TensorRT")
            
        elif config.get('use_coreml'):
            # Export for CoreML (would use coremltools)
            print(f"CoreML export would be performed here")
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            
        elif config.get('use_onnx'):
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input,
                            f"{output_dir}/model.onnx",
                            opset_version=11)
            print(f"Model exported to {output_dir}/model.onnx")
            
        else:
            # Default PyTorch format
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"Model exported to {output_dir}/model.pt")
        
        # Save configuration
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Generate deployment script
        self._generate_deployment_script(config, output_dir)
        
        print(f"Deployment package created in {output_dir}/")
    
    def _generate_deployment_script(self, config: Dict, output_dir: str):
        """Generate platform-specific deployment script"""
        
        script_template = """#!/usr/bin/env python3
\"\"\"
Auto-generated deployment script for {platform}
Generated by TriOpt Framework
\"\"\"

import torch
import numpy as np

# Configuration
CONFIG = {config}

def load_model():
    \"\"\"Load optimized model\"\"\"
    # Model loading code here
    pass

def preprocess(image):
    \"\"\"Preprocess input\"\"\"
    # Preprocessing code here
    pass

def inference(model, input_data):
    \"\"\"Run inference\"\"\"
    with torch.no_grad():
        output = model(input_data)
    return output

def main():
    model = load_model()
    # Deployment code here
    print("Model ready for inference")

if __name__ == "__main__":
    main()
"""
        
        script = script_template.format(
            platform=config['hardware_specs']['platform'],
            config=json.dumps(config, indent=4)
        )
        
        with open(f"{output_dir}/deploy.py", 'w') as f:
            f.write(script)
        
        # Make executable
        os.chmod(f"{output_dir}/deploy.py", 0o755)


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
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Initialize optimizer
    platform_optimizer = PlatformSpecificOptimizer()
    
    # Define constraints
    constraints = {
        'latency': 10.0,      # 10ms target
        'memory': 50.0,       # 50MB budget
        'accuracy': 0.95,     # 95% minimum accuracy
        'max_batch': 32
    }
    
    # Create and optimize model
    model = SimpleModel()
    optimized_model, config = platform_optimizer.optimize(model, constraints)
    
    print("\nOptimization Configuration:")
    print(json.dumps(config, indent=2))
    
    # Export deployment package
    platform_optimizer.export_deployment_package(
        optimized_model, config, "deployment_package"
    )
    
    # Display hardware capabilities
    specs = platform_optimizer.specs
    print(f"\nHardware Capabilities:")
    print(f"  Platform: {specs.platform_name}")
    print(f"  Device Type: {specs.device_type}")
    print(f"  Compute Capability: {specs.compute_capability}")
    print(f"  Memory: {specs.memory_gb:.1f} GB")
    print(f"  Cores: {specs.cores}")
    print(f"  Frequency: {specs.frequency_mhz:.0f} MHz")
    print(f"  Power Budget: {specs.power_budget_watts:.0f} W")
    print(f"  Supported Precisions: {', '.join(specs.supported_precisions)}")
    print(f"  TensorRT: {'Yes' if specs.has_tensorrt else 'No'}")
    print(f"  CoreML: {'Yes' if specs.has_coreml else 'No'}")
    print(f"  ONNX: {'Yes' if specs.has_onnx else 'No'}")
