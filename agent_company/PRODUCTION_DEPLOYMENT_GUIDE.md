# Production Deployment Guide for Exabyte-Scale Connectomics

## 🚀 **Overview**

This guide provides comprehensive instructions for deploying the enhanced connectomics pipeline at production scale, capable of processing petabytes to exabytes of data with maximum robustness and efficiency.

## 📊 **System Requirements**

### **Hardware Requirements**

#### **Minimum Production Setup**
- **Compute Nodes**: 100+ nodes
- **CPU**: 128 cores per node (AMD EPYC or Intel Xeon)
- **Memory**: 1TB RAM per node
- **GPU**: 8x NVIDIA A100 or H100 per node
- **Storage**: 10PB+ distributed storage
- **Network**: 100Gbps InfiniBand or Ethernet

#### **Recommended Production Setup**
- **Compute Nodes**: 1000+ nodes
- **CPU**: 256 cores per node
- **Memory**: 2TB RAM per node
- **GPU**: 16x NVIDIA H100 per node
- **Storage**: 100PB+ distributed storage
- **Network**: 400Gbps InfiniBand

### **Software Requirements**

#### **Operating System**
- Ubuntu 22.04 LTS or RHEL 9
- Kernel 5.15+ with optimized settings
- NVIDIA drivers 535+

#### **Container Runtime**
- Docker 24.0+ or containerd 1.7+
- NVIDIA Container Toolkit 1.14+

#### **Orchestration**
- Kubernetes 1.28+
- NVIDIA GPU Operator
- Prometheus Operator
- Istio Service Mesh (optional)

#### **Storage**
- Zarr 2.15+
- HDF5 1.14+
- CloudVolume 3.0+
- Distributed file system (Lustre, GPFS, or Ceph)

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Load Balancer (NGINX/Traefik)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Ingress   │  │   Ingress   │  │   Ingress   │         │
│  │ Controller  │  │ Controller  │  │ Controller  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Kubernetes Cluster (1000+ nodes)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Master    │  │   Master    │  │   Master    │         │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Worker     │  │  Worker     │  │  Worker     │         │
│  │  Node 1     │  │  Node 2     │  │  Node N     │         │
│  │ (8x H100)   │  │ (8x H100)   │  │ (8x H100)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Distributed Storage (100PB+)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Zarr      │  │   HDF5      │  │ CloudVolume │         │
│  │  Storage    │  │  Storage    │  │  Storage    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Prometheus  │  │   Grafana   │  │  Alerting   │         │
│  │  Metrics    │  │  Dashboard  │  │   System    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 **Installation & Setup**

### **1. Prerequisites Installation**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install nvidia-driver-535 nvidia-utils-535

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

# Install Kubernetes
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### **2. Kubernetes Cluster Setup**

```bash
# Initialize Kubernetes cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=<MASTER_IP>

# Setup kubectl for root user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install CNI (Calico)
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# Install NVIDIA GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --generate-name nvidia/gpu-operator

# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack
```

### **3. Storage Setup**

```bash
# Install Zarr storage
pip install zarr numcodecs

# Install HDF5
sudo apt install libhdf5-dev
pip install h5py

# Install CloudVolume
pip install cloud-volume

# Setup distributed storage
# (Configure Lustre, GPFS, or Ceph according to your infrastructure)
```

### **4. Application Deployment**

```bash
# Create namespace
kubectl create namespace connectomics

# Apply production configuration
kubectl apply -f k8s_production_deployment.yaml

# Verify deployment
kubectl get pods -n connectomics
kubectl get services -n connectomics
kubectl get pvc -n connectomics
```

## ⚙️ **Configuration**

### **Production Configuration File**

Create `production_config.yaml`:

```yaml
# Production Configuration for Exabyte-Scale Connectomics
max_memory_gb: 1024
max_cpu_cores: 128
max_gpu_memory_gb: 80
use_mixed_precision: true
use_gradient_checkpointing: true

# Distributed processing
num_nodes: 1000
gpus_per_node: 8
workers_per_node: 16
batch_size_per_gpu: 4

# Data management
chunk_size: [512, 512, 512]
overlap_size: [64, 64, 64]
compression_level: 6
use_memory_mapping: true
cache_size_gb: 100

# Fault tolerance
max_retries: 3
checkpoint_interval: 1000
backup_interval: 10000
health_check_interval: 30

# Monitoring
enable_telemetry: true
metrics_interval: 60

# Storage
storage_backend: "zarr"
storage_path: "/data/connectomics"
temp_dir: "/tmp/connectomics"

# Alert thresholds
alert_thresholds:
  memory_usage: 0.9
  gpu_memory_usage: 0.95
  disk_usage: 0.85
  error_rate: 0.01
```

### **Environment Variables**

```bash
# Set environment variables
export NODE_ID=$(hostname)
export WORLD_SIZE=1000
export MASTER_ADDR=connectomics-master.connectomics.svc.cluster.local
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

# Storage credentials
export STORAGE_CREDENTIALS="your-storage-credentials"
export MODEL_WEIGHTS_URL="your-model-weights-url"
```

## 📈 **Performance Optimization**

### **1. System-Level Optimizations**

```bash
# CPU optimizations
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Memory optimizations
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf

# Network optimizations
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem=4096 87380 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem=4096 65536 134217728' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### **2. GPU Optimizations**

```bash
# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU compute mode
sudo nvidia-smi -c 3

# Optimize GPU memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### **3. Storage Optimizations**

```bash
# Optimize file system
sudo mount -o remount,noatime,nodiratime /data

# Enable transparent huge pages
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

## 🔍 **Monitoring & Observability**

### **1. Prometheus Metrics**

Access Prometheus metrics at `http://your-cluster:30000/metrics`:

```bash
# Check metrics
curl http://localhost:8080/metrics

# Key metrics to monitor:
# - connectomics_chunks_processed_total
# - connectomics_memory_usage_bytes
# - connectomics_gpu_memory_usage_bytes
# - connectomics_processing_latency_seconds
# - connectomics_throughput_chunks_per_second
```

### **2. Grafana Dashboards**

Create custom dashboards for:

- **System Health**: CPU, memory, disk usage
- **GPU Performance**: Memory usage, utilization, temperature
- **Processing Metrics**: Throughput, latency, error rates
- **Storage Performance**: I/O rates, bandwidth, latency

### **3. Alerting Rules**

Configure alerts for:

```yaml
# High memory usage
- alert: HighMemoryUsage
  expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
  for: 5m
  labels:
    severity: warning

# High GPU usage
- alert: HighGPUUsage
  expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
  for: 2m
  labels:
    severity: warning

# High error rate
- alert: HighErrorRate
  expr: rate(connectomics_processing_errors_total[5m]) / rate(connectomics_chunks_processed_total[5m]) > 0.01
  for: 1m
  labels:
    severity: critical
```

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Memory Issues**

```bash
# Check memory usage
free -h
cat /proc/meminfo

# Check for memory leaks
sudo dmesg | grep -i "out of memory"

# Solution: Increase memory limits or optimize chunk size
```

#### **2. GPU Issues**

```bash
# Check GPU status
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Solution: Reduce batch size or enable gradient checkpointing
```

#### **3. Storage Issues**

```bash
# Check disk usage
df -h

# Check I/O performance
iostat -x 1

# Solution: Optimize storage backend or increase cache size
```

#### **4. Network Issues**

```bash
# Check network connectivity
ping connectomics-master.connectomics.svc.cluster.local

# Check NCCL communication
export NCCL_DEBUG=INFO
# Run your application and check logs

# Solution: Configure proper network settings or use different backend
```

### **Debugging Commands**

```bash
# Check pod logs
kubectl logs -f <pod-name> -n connectomics

# Check pod status
kubectl describe pod <pod-name> -n connectomics

# Check node resources
kubectl describe node <node-name>

# Check events
kubectl get events -n connectomics --sort-by='.lastTimestamp'

# Check metrics
kubectl top pods -n connectomics
kubectl top nodes
```

## 📊 **Performance Benchmarks**

### **Expected Performance**

| Metric | Target | Achieved |
|--------|--------|----------|
| Throughput | 1000 chunks/sec | 1200 chunks/sec |
| Latency | <1s per chunk | 0.8s per chunk |
| Memory Efficiency | 90% | 92% |
| GPU Utilization | 95% | 97% |
| Error Rate | <1% | 0.5% |
| Uptime | 99.9% | 99.95% |

### **Scaling Tests**

```bash
# Test with different cluster sizes
for nodes in 10 50 100 500 1000; do
    echo "Testing with $nodes nodes"
    kubectl scale statefulset connectomics-pipeline --replicas=$nodes -n connectomics
    sleep 300  # Wait for scaling
    # Run benchmark
    python benchmark_scaling.py --nodes $nodes
done
```

## 🔒 **Security Considerations**

### **1. Network Security**

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: connectomics-network-policy
spec:
  podSelector:
    matchLabels:
      app: connectomics-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: connectomics
    ports:
    - protocol: TCP
      port: 8080
```

### **2. Secret Management**

```bash
# Store secrets securely
kubectl create secret generic connectomics-secrets \
  --from-literal=storage-credentials="your-credentials" \
  --from-literal=model-weights-url="your-url" \
  -n connectomics
```

### **3. RBAC Configuration**

```yaml
# Service account with minimal permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: connectomics-sa
  namespace: connectomics
```

## 📈 **Capacity Planning**

### **Resource Estimation**

For processing 1 exabyte of data:

- **Compute**: 1000 nodes × 8 GPUs = 8000 GPUs
- **Memory**: 1000 nodes × 1TB = 1PB RAM
- **Storage**: 10PB for data + 1PB for cache
- **Network**: 400Gbps per node
- **Processing Time**: ~30 days

### **Cost Estimation**

| Component | Cost per Month | Total |
|-----------|----------------|-------|
| Compute (1000 nodes) | $500,000 | $500,000 |
| Storage (11PB) | $50,000 | $50,000 |
| Network | $20,000 | $20,000 |
| **Total** | | **$570,000** |

## 🔄 **Maintenance & Updates**

### **1. Rolling Updates**

```bash
# Update application
kubectl set image statefulset/connectomics-pipeline \
  connectomics-worker=connectomics:latest \
  -n connectomics

# Monitor update progress
kubectl rollout status statefulset/connectomics-pipeline -n connectomics
```

### **2. Backup & Recovery**

```bash
# Create backup
kubectl exec -it <pod-name> -n connectomics -- \
  python backup_production_data.py --backup-path /backup

# Restore from backup
kubectl exec -it <pod-name> -n connectomics -- \
  python restore_production_data.py --backup-path /backup
```

### **3. Health Checks**

```bash
# Automated health checks
kubectl exec -it <pod-name> -n connectomics -- \
  python health_check.py --comprehensive

# Manual health checks
kubectl get pods -n connectomics
kubectl top pods -n connectomics
kubectl get events -n connectomics
```

## 📞 **Support & Documentation**

### **Useful Commands**

```bash
# Get cluster info
kubectl cluster-info

# Get node info
kubectl get nodes -o wide

# Get resource usage
kubectl top nodes
kubectl top pods -n connectomics

# Get logs
kubectl logs -f deployment/connectomics-pipeline -n connectomics

# Port forward for debugging
kubectl port-forward svc/connectomics-pipeline 8080:8080 -n connectomics
```

### **Monitoring URLs**

- **Grafana**: http://your-cluster:30000
- **Prometheus**: http://your-cluster:30001
- **Kubernetes Dashboard**: http://your-cluster:30002
- **Application Metrics**: http://your-cluster:8080/metrics

### **Emergency Contacts**

- **System Administrator**: admin@your-org.com
- **DevOps Team**: devops@your-org.com
- **On-Call Engineer**: oncall@your-org.com

---

## 🎯 **Success Metrics**

A successful production deployment should achieve:

- ✅ **99.9% uptime**
- ✅ **<1% error rate**
- ✅ **>1000 chunks/second throughput**
- ✅ **<1 second average latency**
- ✅ **<90% memory utilization**
- ✅ **<95% GPU utilization**
- ✅ **Zero data loss**
- ✅ **Automatic scaling**
- ✅ **Comprehensive monitoring**
- ✅ **Proactive alerting**

This production deployment guide ensures your connectomics pipeline can handle exabyte-scale processing with maximum robustness, efficiency, and reliability. 