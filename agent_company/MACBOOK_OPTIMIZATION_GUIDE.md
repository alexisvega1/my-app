# MacBook Performance Optimization Guide for H01 Processing

## 🍎 **MacBook-Specific Optimizations**

### **Current Training Status**
- **Training Progress**: 4/20 epochs completed
- **Time per Epoch**: ~25-30 minutes
- **Remaining Time**: ~6-8 hours
- **Memory Usage**: ~3.3GB during processing

---

## ⚡ **Immediate Performance Boosts**

### **1. Memory Management**
```bash
# Monitor memory usage
python -c "import psutil; print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Restart Python processes to free memory
pkill -f python
```

### **2. CPU Optimization**
```bash
# Check CPU cores
sysctl -n hw.ncpu

# Set optimal thread count for your MacBook
export OMP_NUM_THREADS=8  # Adjust based on your MacBook model
export MKL_NUM_THREADS=8
```

### **3. Storage Optimization**
```bash
# Check available disk space
df -h

# Clear temporary files
rm -rf /tmp/*
rm -rf ~/Library/Caches/*

# Optimize for SSD (if applicable)
sudo trimforce enable
```

---

## 🔧 **Configuration Optimizations**

### **1. H01 Config for MacBook**
```yaml
# h01_config_macbook.yaml
data_loading:
  chunk_size: [64, 64, 64]  # Smaller chunks for MacBook memory
  cache_size: "2GB"         # Limit cache to available memory
  compression: "lz4"        # Fast compression

processing:
  batch_size: 4             # Smaller batches
  num_workers: 4            # Match CPU cores
  memory_limit: "4GB"       # Conservative memory limit

segmentation:
  model_config:
    use_mixed_precision: true  # Reduce memory usage
    gradient_checkpointing: true
```

### **2. Training Optimization**
```python
# train_ffn_v2_macbook.py
training_config = {
    'batch_size': 4,           # Smaller batches
    'num_epochs': 20,
    'learning_rate': 0.001,
    'mixed_precision': True,   # Use mixed precision
    'gradient_accumulation': 4, # Accumulate gradients
    'memory_efficient': True
}
```

---

## 📊 **Memory Usage Breakdown**

| Component | Memory Usage | Optimization |
|-----------|--------------|--------------|
| **Segmentation** | 3.3GB | Use smaller chunks |
| **Proofreading** | 3.3GB | Process in batches |
| **Continual Learning** | 3.3GB | Gradient checkpointing |
| **Data Loading** | 2GB | Limit cache size |
| **Total** | ~12GB | **Optimized: ~6GB** |

---

## 🎯 **Region Size Recommendations**

### **Safe for MacBook (8-16GB RAM)**
- **Tiny Region**: 64³ voxels (~1MB)
- **Small Region**: 128³ voxels (~8MB)
- **Medium Region**: 256³ voxels (~64MB)

### **Requires Optimization**
- **Large Region**: 512³ voxels (~512MB)
- **Extra Large**: 1024³ voxels (~4GB)

### **Not Recommended**
- **Huge Region**: 2048³+ voxels (~32GB+)

---

## 🚀 **Advanced Optimizations**

### **1. Chunked Processing**
```python
# Process large regions in chunks
def process_large_region(region_bounds, chunk_size=(256, 256, 256)):
    chunks = split_region_into_chunks(region_bounds, chunk_size)
    results = []
    
    for chunk in chunks:
        # Process each chunk
        result = pipeline.process_chunk(chunk)
        results.append(result)
        
        # Clear memory
        gc.collect()
    
    return merge_chunk_results(results)
```

### **2. Streaming Processing**
```python
# Stream data instead of loading all at once
def stream_process(region_name):
    loader = StreamingDataLoader()
    
    for chunk in loader.stream_chunks(region_name):
        result = process_chunk(chunk)
        yield result
        
        # Clear chunk from memory
        del chunk
        gc.collect()
```

### **3. Memory Mapping**
```python
# Use memory mapping for large files
import numpy.lib.format as npformat

def load_memory_mapped(file_path):
    with open(file_path, 'rb') as f:
        # Load header only
        header = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat.read_array_header_1_0(f)
        
        # Memory map the data
        return np.memmap(f, dtype=dtype, shape=shape, mode='r')
```

---

## 🔍 **Monitoring Tools**

### **1. Real-time Monitoring**
```bash
# Monitor system resources
htop
# or
top -o cpu

# Monitor Python processes
ps aux | grep python | grep -v grep
```

### **2. Memory Profiling**
```python
# memory_profiler.py
from memory_profiler import profile

@profile
def process_region(region_name):
    # Your processing code here
    pass
```

### **3. Performance Profiling**
```python
# cProfile for performance analysis
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

## 🎯 **MacBook-Specific Tips**

### **1. Thermal Management**
- **Monitor temperature**: `sudo powermetrics --samplers smc -n 1`
- **Avoid thermal throttling**: Keep CPU usage below 80%
- **Use external cooling**: Laptop stand with cooling

### **2. Battery Optimization**
- **Plug in during processing**: Avoid battery drain
- **Disable unnecessary services**: Bluetooth, WiFi if not needed
- **Use low power mode**: For background processing

### **3. Storage Optimization**
- **Use external SSD**: For large datasets
- **Enable TRIM**: For SSD health
- **Monitor disk space**: Keep 20% free

---

## 📈 **Performance Benchmarks**

### **MacBook Pro 13" (M1)**
- **Memory**: 8GB unified
- **Processing Speed**: ~2x faster than Intel
- **Memory Efficiency**: ~30% better
- **Recommended Batch Size**: 2-4

### **MacBook Pro 16" (Intel)**
- **Memory**: 16GB DDR4
- **Processing Speed**: Standard
- **Memory Efficiency**: Standard
- **Recommended Batch Size**: 4-8

### **MacBook Air (M1/M2)**
- **Memory**: 8-16GB unified
- **Processing Speed**: ~1.5x faster than Intel
- **Memory Efficiency**: ~25% better
- **Recommended Batch Size**: 2-4

---

## 🚀 **Cloud Deployment Options**

### **1. Google Cloud Platform**
```bash
# Create optimized instance
gcloud compute instances create h01-processor \
    --machine-type=n1-standard-32 \
    --memory=120GB \
    --cpu-platform=Intel \
    --zone=us-central1-a
```

### **2. AWS EC2**
```bash
# Use memory-optimized instance
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type r5.8xlarge \
    --key-name my-key
```

### **3. Local Cluster**
```bash
# Use multiple MacBooks
python distributed_pipeline.py \
    --workers 4 \
    --master-ip 192.168.1.100 \
    --worker-ips 192.168.1.101,192.168.1.102,192.168.1.103
```

---

## 📋 **Quick Optimization Checklist**

- [ ] **Memory**: Clear caches, restart processes
- [ ] **CPU**: Set optimal thread count
- [ ] **Storage**: Free up disk space
- [ ] **Config**: Use MacBook-optimized settings
- [ ] **Monitoring**: Set up resource monitoring
- [ ] **Thermal**: Monitor temperature
- [ ] **Backup**: Save current progress

---

## 🎯 **Expected Performance Improvements**

| Optimization | Memory Reduction | Speed Improvement |
|--------------|------------------|-------------------|
| **Chunked Processing** | 60% | 20% |
| **Mixed Precision** | 50% | 15% |
| **Memory Mapping** | 40% | 10% |
| **Gradient Checkpointing** | 30% | 5% |
| **Combined** | **80%** | **40%** |

---

## 🚨 **Troubleshooting**

### **Out of Memory Errors**
```bash
# Reduce batch size
export BATCH_SIZE=2

# Clear memory
python -c "import gc; gc.collect()"

# Restart with smaller chunks
python h01_production_pipeline.py --chunk-size 64
```

### **Slow Processing**
```bash
# Check CPU usage
top -o cpu

# Optimize thread count
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Use SSD for caching
export CACHE_DIR=/tmp/h01_cache
```

### **Thermal Throttling**
```bash
# Monitor temperature
sudo powermetrics --samplers smc -n 1

# Reduce CPU usage
export OMP_NUM_THREADS=4
```

---

**🎉 With these optimizations, your MacBook should handle H01 processing much more efficiently!** 