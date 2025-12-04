# 🚀 निराक्षण CUDA - Quick Reference Card

## GPU Optimization Commands

### Check GPU Status
```bash
# GPU info
nvidia-smi

# CUDA version
nvcc --version

# GPU temperature & utilization (real-time)
watch -n 1 nvidia-smi
```

### Verify Installation
```python
# TensorFlow GPU
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# OpenCV CUDA
import cv2
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())

# dlib CUDA
import dlib
print("dlib CUDA:", dlib.DLIB_USE_CUDA)

# CuPy
import cupy as cp
print("CuPy version:", cp.__version__)
```

---

## Performance Tuning

### 1. GPU Memory Settings
```python
# Enable memory growth (recommended)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Or set memory limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]  # 6GB
)
```

### 2. Batch Processing
```json
// config.json
{
    "performance": {
        "gpu_batch_size": 4  // Increase for better GPU utilization
    }
}
```

### 3. Frame Skipping
```json
// Disable frame skipping with GPU
{
    "performance": {
        "enable_frame_skipping": false,
        "frame_skip_interval": 1
    }
}
```

---

## Common Issues & Quick Fixes

### ❌ GPU Not Detected
```bash
# Check driver
nvidia-smi

# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0
```

### ❌ Out of Memory
```python
# Reduce batch size
CONFIG["performance"]["gpu_batch_size"] = 2

# Enable memory growth
tf.config.experimental.set_memory_growth(gpus[0], True)
```

### ❌ Low FPS
```bash
# Check GPU utilization
nvidia-smi

# If low (<50%), check:
# 1. Is GPU actually being used?
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"

# 2. OpenCV CUDA enabled?
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### ❌ dlib CUDA Not Working
```bash
# Rebuild dlib with CUDA
cd ~/dlib
mkdir build && cd build
cmake .. -DDLIB_USE_CUDA=1 -DCUDA_ARCH_NAME=Pascal
make -j$(nproc)
cd .. && python setup.py install
```

---

## GPU Monitoring Commands

### Real-time Monitoring
```bash
# Basic monitoring
nvidia-smi -l 1

# Detailed monitoring
nvidia-smi dmon -s pucvmet -d 1

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```

### Performance Logging
```bash
# Log GPU stats to file
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_stats.csv
```

---

## Jupyter Notebook Tips

### Start Jupyter with GPU
```bash
cd ~/niraakshan_cuda
source niraakshan_cuda_env/bin/activate
jupyter notebook
```

### Check GPU in Notebook
```python
# Cell 1: Verify GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")
```

### Monitor GPU During Execution
```bash
# In separate terminal
watch -n 0.5 nvidia-smi
```

---

## Environment Management

### Activate Environment
```bash
cd ~/niraakshan_cuda
source niraakshan_cuda_env/bin/activate
```

### Update Packages
```bash
source niraakshan_cuda_env/bin/activate
pip install --upgrade tensorflow[and-cuda]
pip install --upgrade cupy-cuda11x
```

### Backup Environment
```bash
pip freeze > requirements_cuda.txt
```

### Restore Environment
```bash
pip install -r requirements_cuda.txt
```

---

## Configuration Quick Reference

### Optimal Config for GTX 1080 (8GB)
```json
{
    "detection": {
        "use_gpu": true
    },
    "performance": {
        "gpu_batch_size": 4,
        "frame_skip_interval": 1,
        "enable_frame_skipping": false
    },
    "gpu": {
        "device_id": 0,
        "memory_fraction": 0.8,
        "allow_growth": true
    }
}
```

### Low Memory Config (if OOM errors)
```json
{
    "performance": {
        "gpu_batch_size": 2
    },
    "gpu": {
        "memory_fraction": 0.6,
        "allow_growth": true
    }
}
```

### Maximum Performance Config
```json
{
    "performance": {
        "gpu_batch_size": 8,
        "frame_skip_interval": 0
    },
    "gpu": {
        "memory_fraction": 0.9,
        "allow_growth": false
    }
}
```

---

## Benchmarking

### Test GPU Performance
```python
import time
import numpy as np
import tensorflow as tf

# Warm up GPU
for _ in range(10):
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)

# Benchmark
start = time.time()
for _ in range(100):
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
        tf.reduce_sum(b).numpy()  # Force execution
elapsed = time.time() - start

print(f"GPU Benchmark: {elapsed:.2f}s for 100 iterations")
print(f"Average: {elapsed*10:.2f}ms per iteration")
```

### Expected Results (GTX 1080)
- **GPU Benchmark**: ~0.5-1.0s for 100 iterations
- **Average**: 5-10ms per iteration
- If slower, check GPU utilization with `nvidia-smi`

---

## Emergency Recovery

### Reset GPU
```bash
# Restart NVIDIA driver
sudo systemctl restart nvidia-persistenced

# Or reboot
sudo reboot
```

### Clear GPU Memory
```python
import gc
import tensorflow as tf
import cupy as cp

# Clear everything
gc.collect()
tf.keras.backend.clear_session()

# CuPy memory
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
```

### Reset CUDA Context
```python
# In notebook, restart kernel
# Runtime -> Restart Runtime

# Or in script
import cupy as cp
cp.cuda.runtime.deviceReset()
```

---

## Useful Aliases

Add to `~/.bashrc`:
```bash
# निराक्षण shortcuts
alias nir='cd ~/niraakshan_cuda && source niraakshan_cuda_env/bin/activate'
alias nir-jupyter='nir && jupyter notebook'
alias nir-gpu='watch -n 1 nvidia-smi'
alias nir-temp='nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -l 1'
```

---

## Performance Checklist

Before starting:
- [ ] GPU driver installed (`nvidia-smi` works)
- [ ] CUDA toolkit installed (`nvcc --version` works)
- [ ] Environment activated
- [ ] TensorFlow detects GPU
- [ ] OpenCV CUDA enabled
- [ ] dlib CUDA enabled (optional, for best performance)
- [ ] GPU temperature < 85°C

During execution:
- [ ] GPU utilization > 80% (`nvidia-smi`)
- [ ] FPS > 20 (check in logs)
- [ ] Memory usage < 7GB (leave 1GB buffer)
- [ ] No memory errors in logs

---

## Contact & Support

### Check Logs
```bash
# Session log
tail -f ~/niraakshan_cuda/logs/session_cuda.log

# Event log
tail -f ~/niraakshan_cuda/logs/events.csv
```

### Report Issue
Include:
1. `nvidia-smi` output
2. `nvcc --version` output
3. Python version
4. Last 50 lines of `session_cuda.log`

---

## Quick Start (Copy-Paste)

```bash
# 1. Navigate to project
cd ~/niraakshan_cuda

# 2. Activate environment
source niraakshan_cuda_env/bin/activate

# 3. Check GPU
nvidia-smi

# 4. Start Jupyter
jupyter notebook

# 5. In another terminal, monitor GPU
watch -n 1 nvidia-smi
```

---

**Version**: 1.0  
**Last Updated**: December 2025  
**GPU**: NVIDIA GTX 1080  
**OS**: Ubuntu 22.04.05 LTS
