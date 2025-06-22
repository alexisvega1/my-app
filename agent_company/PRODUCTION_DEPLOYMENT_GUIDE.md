# Production Deployment Guide

## 🚀 **Overview**

This guide covers deploying the enhanced connectomics pipeline to production environments with enterprise-grade reliability, scalability, and monitoring.

## 📋 **Prerequisites**

### **Hardware Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (V100, A100, RTX 3090+)
- **CPU**: 16+ cores recommended
- **RAM**: 32GB+ system memory
- **Storage**: 1TB+ SSD for data and checkpoints
- **Network**: High-speed internet for data transfer

### **Software Requirements**
- **OS**: Ubuntu 20.04+ or CentOS 8+
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Kubernetes**: 1.24+ (for K8s deployment)
- **NVIDIA Drivers**: 470+ with CUDA 11.8+
- **Python**: 3.9+

## 🏗️ **Deployment Options**

### **Option 1: Docker Compose (Recommended for Small-Medium Scale)**

```bash
# Clone repository
git clone <your-repo>
cd my-app

# Build and start services
docker-compose -f agent_company/deployment/docker-compose.yml up -d

# Check status
docker-compose ps
```

### **Option 2: Kubernetes (Recommended for Large Scale)**

```bash
# Apply Kubernetes manifests
kubectl apply -f agent_company/deployment/k8s/

# Check deployment status
kubectl get pods -l app=connectomics-pipeline
kubectl get services
```

### **Option 3: Bare Metal Deployment**

```bash
# Install dependencies
pip install -r agent_company/requirements-production.txt

# Run pipeline
python -m agent_company.enhanced_pipeline --environment production
```

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Required
export ENVIRONMENT=production
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/model.pt

# Optional
export LOG_LEVEL=INFO
export DATA_PATH=/path/to/data
export CHECKPOINT_PATH=/path/to/checkpoints
```

### **Configuration File**

Create `production_config.yaml`:

```yaml
environment: production
data:
  batch_size: 4
  num_workers: 8
  prefetch_factor: 2
  cache_dir: /app/cache
  data_path: /app/data

model:
  input_channels: 1
  output_channels: 1
  hidden_channels: 64
  depth: 4

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 10
  gradient_clip_val: 1.0

monitoring:
  prometheus_port: 9090
  metrics_interval: 30
  health_check_interval: 60

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

## 📊 **Monitoring Setup**

### **Prometheus Configuration**

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'connectomics-pipeline'
    static_configs:
      - targets: ['connectomics-pipeline:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'model-server'
    static_configs:
      - targets: ['model-server:8000']
    metrics_path: '/metrics'
```

### **Grafana Dashboard**

1. Access Grafana at `http://localhost:3000`
2. Login with `admin/admin`
3. Import dashboard from `agent_company/deployment/grafana/dashboards/`

## 🔒 **Security Considerations**

### **Network Security**
- Use HTTPS/TLS for all external communications
- Implement proper firewall rules
- Use VPN for remote access

### **Authentication & Authorization**
- Implement API key authentication
- Use role-based access control (RBAC)
- Enable audit logging

### **Data Security**
- Encrypt data at rest and in transit
- Implement data backup and recovery
- Use secure storage solutions

## 📈 **Scaling Strategies**

### **Horizontal Scaling**
```bash
# Scale Kubernetes deployment
kubectl scale deployment connectomics-pipeline --replicas=3

# Scale Docker Compose services
docker-compose up -d --scale connectomics-pipeline=3
```

### **Vertical Scaling**
- Increase GPU resources
- Add more CPU cores and RAM
- Use faster storage (NVMe SSDs)

### **Load Balancing**
```nginx
# Nginx configuration for load balancing
upstream connectomics_backend {
    server connectomics-pipeline-1:8000;
    server connectomics-pipeline-2:8000;
    server connectomics-pipeline-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://connectomics_backend;
    }
}
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **GPU Memory Issues**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=2
   
   # Monitor GPU usage
   nvidia-smi
   ```

2. **Out of Memory Errors**
   ```bash
   # Increase swap space
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Slow Training**
   ```bash
   # Check data loading
   export NUM_WORKERS=0  # Debug data loading
   
   # Profile performance
   python -m agent_company.monitoring --profile
   ```

### **Log Analysis**
```bash
# View logs
docker-compose logs -f connectomics-pipeline

# Search for errors
grep -i error logs/pipeline.log

# Monitor resource usage
htop
nvidia-smi -l 1
```

## 🔄 **Backup & Recovery**

### **Data Backup**
```bash
# Backup data directory
tar -czf data_backup_$(date +%Y%m%d).tar.gz /app/data

# Backup checkpoints
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz /app/checkpoints
```

### **Disaster Recovery**
```bash
# Restore from backup
tar -xzf data_backup_20231201.tar.gz -C /app/

# Restart services
docker-compose restart
```

## 📞 **Support & Maintenance**

### **Regular Maintenance**
- Monitor disk space usage
- Update dependencies monthly
- Review and rotate logs
- Check security updates

### **Performance Optimization**
- Profile training and inference
- Optimize data loading pipeline
- Tune hyperparameters
- Monitor resource utilization

### **Contact Information**
- **Technical Support**: [your-email]
- **Documentation**: [docs-url]
- **Issues**: [github-issues-url]

## 🎯 **Next Steps**

1. **Set up monitoring dashboards**
2. **Configure alerting rules**
3. **Implement automated backups**
4. **Set up CI/CD pipeline**
5. **Plan for scaling**

---

*For additional support, refer to the [README_ENHANCED.md](README_ENHANCED.md) or create an issue in the repository.* 