"""
System-Level Online Adaptation Module
Implements dynamic batch size optimization, thread configuration, and precision selection
"""

import torch
import numpy as np
import time
import psutil
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import multiprocessing as mp

@dataclass
class SystemMetrics:
    """System performance metrics"""
    latency: float          # ms
    throughput: float       # images/sec
    memory_usage: float     # MB
    cpu_utilization: float  # percentage
    gpu_utilization: float  # percentage
    power_consumption: float # Watts (estimated)

class DynamicBatchOptimizer:
    """Dynamic batch size optimization with online learning"""
    
    def __init__(self, initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 256,
                 learning_rate: float = 0.01):
        """
        Initialize batch optimizer
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            learning_rate: Learning rate for online updates
        """
        self.current_batch_size = initial_batch_size
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.learning_rate = learning_rate
        
        # Performance history
        self.history = deque(maxlen=100)
        self.update_counter = 0
        
    def measure_performance(self, model: torch.nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int,
                           device: str = 'cpu') -> Tuple[float, float]:
        """
        Measure throughput and latency for given batch size
        
        Args:
            model: Model to evaluate
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size to test
            device: Device to run on
        
        Returns:
            Tuple of (throughput, latency)
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure latency
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.perf_counter()
        
        num_iterations = 10
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency = (total_time / num_iterations) * 1000  # ms
        throughput = (batch_size * num_iterations) / total_time  # images/sec
        
        return throughput, latency
    
    def compute_efficiency_score(self, throughput: float, 
                                latency: float,
                                epsilon: float = 1e-6) -> float:
        """
        Compute efficiency score balancing throughput and latency
        
        Args:
            throughput: Images per second
            latency: Latency in ms
            epsilon: Small constant to prevent division by zero
        
        Returns:
            Efficiency score
        """
        # Efficiency = throughput / (latency + epsilon)
        # Higher throughput and lower latency give higher score
        efficiency = throughput / (latency + epsilon)
        return efficiency
    
    def update_batch_size(self, model: torch.nn.Module,
                         input_shape: Tuple[int, ...],
                         device: str = 'cpu') -> int:
        """
        Update batch size based on online learning
        
        Args:
            model: Model to optimize for
            input_shape: Input shape
            device: Device to run on
        
        Returns:
            Updated batch size
        """
        self.update_counter += 1
        
        # Only update every N iterations
        if self.update_counter % 100 != 0:
            return self.current_batch_size
        
        # Measure current performance
        curr_throughput, curr_latency = self.measure_performance(
            model, input_shape, self.current_batch_size, device
        )
        curr_efficiency = self.compute_efficiency_score(curr_throughput, curr_latency)
        
        # Try adjacent batch sizes
        candidates = []
        
        # Try smaller batch size
        if self.current_batch_size > self.min_batch:
            smaller_batch = max(self.min_batch, int(self.current_batch_size * 0.8))
            s_throughput, s_latency = self.measure_performance(
                model, input_shape, smaller_batch, device
            )
            s_efficiency = self.compute_efficiency_score(s_throughput, s_latency)
            candidates.append((smaller_batch, s_efficiency))
        
        # Try larger batch size
        if self.current_batch_size < self.max_batch:
            larger_batch = min(self.max_batch, int(self.current_batch_size * 1.2))
            
            # Check memory constraint
            try:
                l_throughput, l_latency = self.measure_performance(
                    model, input_shape, larger_batch, device
                )
                l_efficiency = self.compute_efficiency_score(l_throughput, l_latency)
                candidates.append((larger_batch, l_efficiency))
            except RuntimeError as e:  # Out of memory
                if "out of memory" in str(e).lower():
                    self.max_batch = self.current_batch_size
        
        # Add current batch size
        candidates.append((self.current_batch_size, curr_efficiency))
        
        # Select best batch size
        best_batch, best_efficiency = max(candidates, key=lambda x: x[1])
        
        # Update with gradient-based learning
        if best_batch != self.current_batch_size:
            gradient = (best_efficiency - curr_efficiency) * np.sign(best_batch - self.current_batch_size)
            self.current_batch_size = int(
                self.current_batch_size + self.learning_rate * gradient * self.current_batch_size
            )
            self.current_batch_size = np.clip(
                self.current_batch_size, self.min_batch, self.max_batch
            )
        
        # Store in history
        self.history.append({
            'batch_size': self.current_batch_size,
            'throughput': curr_throughput,
            'latency': curr_latency,
            'efficiency': curr_efficiency
        })
        
        return self.current_batch_size


class WorkStealingThreadPool:
    """Thread pool with work-stealing for load balancing"""
    
    def __init__(self, num_threads: int = None):
        """
        Initialize thread pool
        
        Args:
            num_threads: Number of worker threads (default: CPU count)
        """
        self.num_threads = num_threads or mp.cpu_count()
        self.threads = []
        self.queues = [queue.Queue(maxsize=64) for _ in range(self.num_threads)]
        self.steal_threshold = 0.2  # Steal when queue load difference > 20%
        self.running = False
        self.results = queue.Queue()
        
    def worker(self, worker_id: int):
        """
        Worker thread with work-stealing
        
        Args:
            worker_id: Thread identifier
        """
        local_queue = self.queues[worker_id]
        
        while self.running:
            task = None
            
            # Try to get task from local queue
            try:
                task = local_queue.get(timeout=0.01)
            except queue.Empty:
                # Try to steal from other queues
                task = self.steal_work(worker_id)
            
            if task is not None:
                func, args = task
                try:
                    result = func(*args)
                    self.results.put((worker_id, result))
                except Exception as e:
                    self.results.put((worker_id, e))
    
    def steal_work(self, worker_id: int) -> Optional[Tuple]:
        """
        Steal work from other threads
        
        Args:
            worker_id: Current worker ID
        
        Returns:
            Stolen task or None
        """
        # Find the queue with most items
        max_queue_id = -1
        max_queue_size = 0
        
        for i, q in enumerate(self.queues):
            if i != worker_id:
                size = q.qsize()
                if size > max_queue_size:
                    max_queue_size = size
                    max_queue_id = i
        
        # Steal if imbalance detected
        if max_queue_id >= 0 and max_queue_size > 0:
            local_size = self.queues[worker_id].qsize()
            
            if (max_queue_size - local_size) / max(max_queue_size, 1) > self.steal_threshold:
                try:
                    return self.queues[max_queue_id].get_nowait()
                except queue.Empty:
                    pass
        
        return None
    
    def start(self):
        """Start worker threads"""
        self.running = True
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker, args=(i,))
            thread.start()
            self.threads.append(thread)
    
    def stop(self):
        """Stop worker threads"""
        self.running = False
        for thread in self.threads:
            thread.join()
        self.threads.clear()
    
    def submit(self, func: Callable, *args):
        """
        Submit task to thread pool
        
        Args:
            func: Function to execute
            args: Function arguments
        """
        # Simple round-robin distribution
        min_queue_id = min(range(self.num_threads), key=lambda i: self.queues[i].qsize())
        self.queues[min_queue_id].put((func, args))
    
    def get_load_balance(self) -> Dict[int, int]:
        """Get current load distribution across threads"""
        return {i: q.qsize() for i, q in enumerate(self.queues)}


class PrecisionModeSelector:
    """Hardware-aware precision mode selection"""
    
    def __init__(self, supported_precisions: List[str] = None):
        """
        Initialize precision selector
        
        Args:
            supported_precisions: List of supported precision modes
        """
        # Detect hardware capabilities
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
                        latency_target: float,
                        epsilon: float = 1e-6) -> str:
        """
        Select optimal precision mode
        
        Args:
            accuracy_requirement: Minimum accuracy (0-1)
            latency_target: Target latency in ms
            epsilon: Small constant
        
        Returns:
            Selected precision mode
        """
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
            score = speedup / (accuracy_loss + epsilon)
            
            if score > best_score:
                best_score = score
                best_precision = precision
        
        self.current_precision = best_precision
        return best_precision
    
    def apply_precision(self, model: torch.nn.Module, 
                       precision: str) -> torch.nn.Module:
        """
        Apply precision mode to model
        
        Args:
            model: Model to convert
            precision: Target precision
        
        Returns:
            Converted model
        """
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
            # In practice, use torch.quantization
            model_int8 = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            return model_int8
        
        elif precision == 'int4':
            # INT4 requires special handling
            print("INT4 quantization requires custom implementation")
            return model
        
        else:
            print(f"Unknown precision: {precision}")
            return model


class SystemLevelAdapter:
    """Main system-level adaptation controller"""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize system adapter
        
        Args:
            model: Model to optimize
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.batch_optimizer = DynamicBatchOptimizer()
        self.thread_pool = WorkStealingThreadPool()
        self.precision_selector = PrecisionModeSelector()
        
        # System monitoring
        self.metrics_history = deque(maxlen=1000)
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
        
        # GPU metrics (if available)
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
            except:
                pass
        
        # Estimate power (simplified)
        power = 10.0 + (cpu_percent / 100) * 50.0  # 10W idle, up to 60W
        if gpu_percent > 0:
            power += (gpu_percent / 100) * 100.0  # GPU adds up to 100W
        
        return SystemMetrics(
            latency=0.0,  # Will be measured separately
            throughput=0.0,  # Will be measured separately
            memory_usage=memory_used_mb,
            cpu_utilization=cpu_percent,
            gpu_utilization=gpu_percent,
            power_consumption=power
        )
    
    def optimize_configuration(self, input_shape: Tuple[int, ...],
                              constraints: Dict[str, float]) -> Dict[str, any]:
        """
        Optimize system configuration
        
        Args:
            input_shape: Model input shape
            constraints: Performance constraints
        
        Returns:
            Optimized configuration
        """
        print("Optimizing system configuration...")
        
        # Update batch size
        optimal_batch = self.batch_optimizer.update_batch_size(
            self.model, input_shape, str(self.device)
        )
        
        # Select precision
        accuracy_req = constraints.get('accuracy', 0.95)
        latency_target = constraints.get('latency', 100.0)
        
        optimal_precision = self.precision_selector.select_precision(
            accuracy_req, latency_target
        )
        
        # Configure threads
        optimal_threads = mp.cpu_count()
        if constraints.get('power', float('inf')) < 50.0:
            # Reduce threads for power constraints
            optimal_threads = max(1, optimal_threads // 2)
        
        config = {
            'batch_size': optimal_batch,
            'precision': optimal_precision,
            'num_threads': optimal_threads,
            'device': str(self.device)
        }
        
        print(f"Optimal configuration: {config}")
        
        return config
    
    def apply_configuration(self, config: Dict[str, any]):
        """
        Apply system configuration
        
        Args:
            config: Configuration to apply
        """
        # Apply precision
        precision = config.get('precision', 'fp32')
        self.model = self.precision_selector.apply_precision(self.model, precision)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Configure threads
        num_threads = config.get('num_threads', mp.cpu_count())
        torch.set_num_threads(num_threads)
        
        print(f"Configuration applied: {config}")
    
    def benchmark(self, input_shape: Tuple[int, ...],
                 batch_size: int = 32,
                 num_iterations: int = 100) -> SystemMetrics:
        """
        Benchmark current configuration
        
        Args:
            input_shape: Input shape
            batch_size: Batch size
            num_iterations: Number of iterations
        
        Returns:
            Performance metrics
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Measure performance
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency = (total_time / num_iterations) * 1000  # ms
        throughput = (batch_size * num_iterations) / total_time
        
        # Get system metrics
        metrics = self.get_system_metrics()
        metrics.latency = latency
        metrics.throughput = throughput
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        if not self.metrics_history:
            return {}
        
        latencies = [m.latency for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        stats = {
            'latency': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            },
            'throughput': {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'max': np.max(throughputs)
            },
            'memory': {
                'mean': np.mean(memory_usage),
                'peak': np.max(memory_usage)
            }
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.fc = torch.nn.Linear(128 * 8 * 8, 10)
            
        def forward(self, x):
            x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
            x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Initialize model and adapter
    model = SimpleModel()
    adapter = SystemLevelAdapter(model)
    
    # Define constraints
    constraints = {
        'latency': 50.0,    # 50ms target
        'accuracy': 0.95,   # 95% accuracy requirement
        'memory': 100.0,    # 100MB memory budget
        'power': 30.0       # 30W power budget
    }
    
    # Optimize configuration
    input_shape = (3, 32, 32)
    config = adapter.optimize_configuration(input_shape, constraints)
    
    # Apply configuration
    adapter.apply_configuration(config)
    
    # Benchmark
    metrics = adapter.benchmark(input_shape, batch_size=config['batch_size'])
    
    print("\nBenchmark Results:")
    print(f"  Latency: {metrics.latency:.2f} ms")
    print(f"  Throughput: {metrics.throughput:.2f} img/s")
    print(f"  Memory: {metrics.memory_usage:.2f} MB")
    print(f"  CPU Usage: {metrics.cpu_utilization:.1f}%")
    print(f"  GPU Usage: {metrics.gpu_utilization:.1f}%")
    print(f"  Power: {metrics.power_consumption:.1f} W")
    
    # Get statistics after multiple runs
    for _ in range(10):
        adapter.benchmark(input_shape, batch_size=config['batch_size'])
    
    stats = adapter.get_statistics()
    print("\nPerformance Statistics:")
    print(f"  Latency: {stats['latency']['mean']:.2f} ± {stats['latency']['std']:.2f} ms")
    print(f"  P95 Latency: {stats['latency']['p95']:.2f} ms")
    print(f"  P99 Latency: {stats['latency']['p99']:.2f} ms")
    print(f"  Throughput: {stats['throughput']['mean']:.2f} ± {stats['throughput']['std']:.2f} img/s")
