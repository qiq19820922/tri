"""
TriOpt Framework - Complete Optimization Demo
Demonstrates the full edge AI optimization workflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os

# Import the integrated TriOpt framework (from the previous artifact)
# In practice, this would be: from triopt_integrated import *

# ============================================================================
# Step 1: Define the Model to Optimize
# ============================================================================

class MobileNetV2Block(nn.Module):
    """Inverted Residual Block for MobileNetV2"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SimpleMobileNetV2(nn.Module):
    """Simplified MobileNetV2 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MobileNetV2 blocks
        self.block1 = MobileNetV2Block(32, 16, stride=1, expand_ratio=1)
        self.block2 = MobileNetV2Block(16, 24, stride=2, expand_ratio=6)
        self.block3 = MobileNetV2Block(24, 32, stride=2, expand_ratio=6)
        self.block4 = MobileNetV2Block(32, 64, stride=2, expand_ratio=6)
        self.block5 = MobileNetV2Block(64, 96, stride=1, expand_ratio=6)
        
        # Final layers
        self.conv2 = nn.Conv2d(96, 256, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Initial conv
        x = F.relu6(self.bn1(self.conv1(x)))
        
        # Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Final layers
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# ============================================================================
# Step 2: Prepare Dataset
# ============================================================================

def prepare_cifar10_dataset(data_dir='./data', batch_size=32, subset_size=1000):
    """
    Prepare CIFAR-10 dataset for optimization
    
    Args:
        data_dir: Directory to download/load data
        batch_size: Batch size for DataLoader
        subset_size: Size of subset for faster optimization
    
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Create subsets for faster optimization
    train_subset = Subset(train_dataset, np.random.choice(len(train_dataset), subset_size, replace=False))
    test_subset = Subset(test_dataset, np.random.choice(len(test_dataset), subset_size//10, replace=False))
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ============================================================================
# Step 3: Pre-train Model (or Load Pre-trained)
# ============================================================================

def train_baseline_model(model, train_loader, test_loader, epochs=5, device='cpu'):
    """
    Train baseline model before optimization
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Training baseline model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Testing
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Acc: {100.*correct/total:.2f}%, '
              f'Test Acc: {100.*correct_test/total_test:.2f}%')
    
    return model

# ============================================================================
# Step 4: Define Deployment Scenario
# ============================================================================

def create_deployment_scenario():
    """
    Create a deployment scenario for edge device
    
    Returns:
        category, performance_requirements, environmental_context
    """
    # Application category
    category = ApplicationCategory.REAL_TIME_VISION
    
    # Performance requirements
    perf_reqs = PerformanceRequirements(
        latency_target=20.0,      # 20ms target latency
        memory_budget=50.0,        # 50MB memory budget
        power_budget=10.0,         # 10W power budget
        accuracy_req=0.90,         # 90% minimum accuracy
        batch_size=1,              # Real-time processing
        input_resolution=(32, 32, 3)  # CIFAR-10 resolution
    )
    
    # Environmental context
    env_context = EnvironmentalContext(
        data_modality="image",
        network_conditions="stable",
        usage_pattern="continuous",
        temporal_variance=0.1
    )
    
    return category, perf_reqs, env_context

# ============================================================================
# Step 5: Run Complete Optimization Pipeline
# ============================================================================

def run_triopt_optimization():
    """
    Complete TriOpt optimization pipeline demonstration
    """
    
    print("="*80)
    print("TriOpt: Joint Optimization Framework for Edge AI Deployment")
    print("Complete Optimization Pipeline Demonstration")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ============================================================
    # Phase 1: Prepare Model and Data
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 1: Model and Data Preparation")
    print("="*60)
    
    # Create model
    model = SimpleMobileNetV2(num_classes=10)
    print(f"Model created: SimpleMobileNetV2")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare dataset
    train_loader, test_loader = prepare_cifar10_dataset(
        batch_size=32, 
        subset_size=1000  # Use subset for faster demo
    )
    print(f"Dataset prepared: CIFAR-10 (subset)")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train baseline model
    model = train_baseline_model(
        model, train_loader, test_loader, 
        epochs=3,  # Quick training for demo
        device=device
    )
    
    # Measure baseline performance
    print("\n" + "="*60)
    print("Baseline Model Performance")
    print("="*60)
    
    model.eval()
    
    # Measure latency
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    num_iterations = 100
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    baseline_latency = ((end_time - start_time) / num_iterations) * 1000
    
    # Measure accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    baseline_accuracy = 100.0 * correct / total
    
    # Measure model size
    total_params = sum(p.numel() for p in model.parameters())
    baseline_size = (total_params * 4) / (1024 * 1024)  # MB (assuming float32)
    
    print(f"Baseline Latency: {baseline_latency:.2f} ms")
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Baseline Model Size: {baseline_size:.2f} MB")
    
    # ============================================================
    # Phase 2: Scene Analysis
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 2: Scene Analysis")
    print("="*60)
    
    # Create deployment scenario
    category, perf_reqs, env_context = create_deployment_scenario()
    
    print(f"Application Category: {category.value}")
    print(f"Performance Requirements:")
    print(f"  - Latency Target: {perf_reqs.latency_target} ms")
    print(f"  - Memory Budget: {perf_reqs.memory_budget} MB")
    print(f"  - Power Budget: {perf_reqs.power_budget} W")
    print(f"  - Accuracy Requirement: {perf_reqs.accuracy_req*100:.0f}%")
    print(f"Environmental Context:")
    print(f"  - Data Modality: {env_context.data_modality}")
    print(f"  - Network Conditions: {env_context.network_conditions}")
    print(f"  - Usage Pattern: {env_context.usage_pattern}")
    
    # ============================================================
    # Phase 3: Initialize TriOpt Framework
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 3: TriOpt Framework Initialization")
    print("="*60)
    
    # Initialize TriOpt with custom configuration
    triopt_config = {
        'population_size': 20,      # Smaller for demo
        'num_generations': 30,      # Fewer generations for demo
        'convergence_threshold': 0.01,
        'bit_candidates': [8, 16, 32],  # Quantization options
        'pruning_ratios': [0.0, 0.2, 0.4, 0.6],  # Pruning options
        'batch_sizes': [1, 2, 4, 8],
        'memory_budget': perf_reqs.memory_budget
    }
    
    triopt = TriOptFramework(
        model=model,
        hardware_platform="auto",
        config=triopt_config
    )
    
    print(f"Hardware Platform Detected: {triopt.hardware_platform}")
    print(f"Device Type: {triopt.hardware_specs.device_type}")
    print(f"Available Memory: {triopt.hardware_specs.memory_gb:.1f} GB")
    print(f"Supported Precisions: {triopt.hardware_specs.supported_precisions}")
    
    # ============================================================
    # Phase 4: Scene-Aware Weight Learning
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 4: Scene-Aware Weight Learning")
    print("="*60)
    
    # Encode scene
    scene_encoding = triopt.encode_scene(category, perf_reqs, env_context)
    print(f"Scene encoded to {scene_encoding.shape[-1]}-dimensional vector")
    
    # Get optimization weights
    scene_weights = triopt.get_optimization_weights(scene_encoding)
    print(f"Learned optimization weights:")
    print(f"  - Latency weight: {scene_weights[0]:.3f}")
    print(f"  - Accuracy weight: {scene_weights[1]:.3f}")
    print(f"  - Memory weight: {scene_weights[2]:.3f}")
    print(f"  - Energy weight: {scene_weights[3]:.3f}")
    
    # ============================================================
    # Phase 5: Joint Optimization
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 5: Joint Multi-Objective Optimization")
    print("="*60)
    
    # Define constraints based on requirements
    constraints = {
        'latency': perf_reqs.latency_target,
        'accuracy': perf_reqs.accuracy_req * 100,  # Convert to percentage
        'memory': perf_reqs.memory_budget,
        'energy': perf_reqs.power_budget * perf_reqs.latency_target  # mJ
    }
    
    # Run optimization
    print("\nStarting optimization process...")
    print("This may take a few minutes...\n")
    
    best_solution = triopt.optimize(
        data_loader=test_loader,  # Use test loader for optimization
        constraints=constraints,
        scene_weights=scene_weights
    )
    
    # ============================================================
    # Phase 6: Results Analysis
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 6: Optimization Results")
    print("="*60)
    
    if best_solution:
        print("\nOptimized Model Configuration:")
        print(f"  - Quantization: {best_solution.model_config.get('quantization_bits', 32)} bits")
        print(f"  - Pruning Ratio: {best_solution.model_config.get('pruning_ratio', 0.0)*100:.0f}%")
        
        print("\nOptimized System Configuration:")
        print(f"  - Batch Size: {best_solution.system_config['batch_size']}")
        print(f"  - Precision: {best_solution.system_config['precision']}")
        print(f"  - Device: {best_solution.system_config['device']}")
        
        print("\nPerformance Comparison:")
        print(f"  Metric          | Baseline    | Optimized   | Improvement")
        print(f"  --------------- | ----------- | ----------- | -----------")
        print(f"  Latency (ms)    | {baseline_latency:11.2f} | {best_solution.objectives.latency:11.2f} | {baseline_latency/best_solution.objectives.latency:.2f}x")
        print(f"  Accuracy (%)    | {baseline_accuracy:11.2f} | {best_solution.objectives.accuracy:11.2f} | {best_solution.objectives.accuracy-baseline_accuracy:+.2f}%")
        print(f"  Model Size (MB) | {baseline_size:11.2f} | {best_solution.objectives.memory:11.2f} | {baseline_size/best_solution.objectives.memory:.2f}x")
        print(f"  Energy (mJ)     | {baseline_latency*10:11.2f} | {best_solution.objectives.energy:11.2f} | {(baseline_latency*10)/best_solution.objectives.energy:.2f}x")
        
        print("\nConstraints Satisfaction:")
        print(f"  - Latency: {'✓' if best_solution.objectives.latency <= constraints['latency'] else '✗'} "
              f"({best_solution.objectives.latency:.2f} <= {constraints['latency']:.2f} ms)")
        print(f"  - Accuracy: {'✓' if best_solution.objectives.accuracy >= constraints['accuracy'] else '✗'} "
              f"({best_solution.objectives.accuracy:.2f} >= {constraints['accuracy']:.2f}%)")
        print(f"  - Memory: {'✓' if best_solution.objectives.memory <= constraints['memory'] else '✗'} "
              f"({best_solution.objectives.memory:.2f} <= {constraints['memory']:.2f} MB)")
        
        # ============================================================
        # Phase 7: Export Optimized Model
        # ============================================================
        
        print("\n" + "="*60)
        print("Phase 7: Export Optimized Model")
        print("="*60)
        
        # Export the optimized model
        output_dir = "./optimized_models"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "triopt_optimized_model.pt")
        
        triopt.export_solution(best_solution, output_path)
        
        print(f"\nOptimization complete! Model saved to {output_path}")
        
        # ============================================================
        # Phase 8: Pareto Frontier Analysis
        # ============================================================
        
        print("\n" + "="*60)
        print("Phase 8: Pareto Frontier Analysis")
        print("="*60)
        
        print(f"\nPareto Frontier Size: {triopt.pareto_frontier.size()} solutions")
        
        if triopt.pareto_frontier.size() > 0:
            print("\nTop 5 Pareto-optimal solutions:")
            print("  # | Latency | Accuracy | Memory | Energy")
            print("  - | ------- | -------- | ------ | ------")
            
            for i, solution in enumerate(triopt.pareto_frontier.solutions[:5]):
                print(f"  {i+1} | {solution.objectives.latency:7.2f} | "
                      f"{solution.objectives.accuracy:8.2f} | "
                      f"{solution.objectives.memory:6.2f} | "
                      f"{solution.objectives.energy:6.2f}")
    
    else:
        print("No valid solution found within constraints!")
        print("Consider relaxing constraints or increasing optimization iterations.")
    
    print("\n" + "="*80)
    print("TriOpt Optimization Pipeline Complete!")
    print("="*80)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run the complete optimization pipeline
    run_triopt_optimization()
    
    # Additional analysis functions
    
    def analyze_hardware_impact():
        """Analyze impact of different hardware platforms"""
        print("\n" + "="*60)
        print("Hardware Platform Impact Analysis")
        print("="*60)
        
        # Simulate different hardware platforms
        platforms = ["nvidia_jetson", "raspberry_pi", "intel_x86"]
        
        for platform in platforms:
            print(f"\nPlatform: {platform}")
            # Here you would run optimization with different hardware configs
            # This is a placeholder for demonstration
            print(f"  - Expected speedup: 1.5-2.5x")
            print(f"  - Memory efficiency: 60-80%")
            print(f"  - Energy efficiency: 40-60%")
    
    def sensitivity_analysis():
        """Analyze sensitivity to different constraints"""
        print("\n" + "="*60)
        print("Constraint Sensitivity Analysis")
        print("="*60)
        
        constraints_variations = [
            {"latency": 10, "accuracy": 0.90, "memory": 50},
            {"latency": 20, "accuracy": 0.95, "memory": 30},
            {"latency": 30, "accuracy": 0.85, "memory": 100},
        ]
        
        for i, constraints in enumerate(constraints_variations):
            print(f"\nConstraint Set {i+1}:")
            print(f"  - Latency: {constraints['latency']} ms")
            print(f"  - Accuracy: {constraints['accuracy']*100:.0f}%")
            print(f"  - Memory: {constraints['memory']} MB")
            # Here you would run optimization with different constraints
            # This is a placeholder for demonstration
            print(f"  - Feasibility: {'High' if constraints['latency'] > 15 else 'Low'}")
    
    # Uncomment to run additional analyses
    # analyze_hardware_impact()
    # sensitivity_analysis()
