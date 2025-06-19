# Cloud Deployment Guide for H01 Processing

## ☁️ **Cloud Deployment Options for Large-Scale H01 Processing**

### **Why Cloud Deployment?**
- **Large Regions**: Process 25GB+ regions that exceed MacBook memory
- **Parallel Processing**: Use multiple machines for faster processing
- **Cost Effective**: Pay only for compute time used
- **Scalability**: Scale up/down based on processing needs

---

## 🚀 **Google Cloud Platform (Recommended)**

### **1. Setup Google Cloud**
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

### **2. Create Optimized Instance**
```bash
# Create high-memory instance for H01 processing
gcloud compute instances create h01-processor \
    --machine-type=n1-standard-32 \
    --memory=120GB \
    --cpu-platform=Intel \
    --zone=us-central1-a \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-ssd \
    --metadata-from-file startup-script=setup_h01.sh
```

### **3. Startup Script (setup_h01.sh)**
```bash
#!/bin/bash
# Install Python and dependencies
apt-get update
apt-get install -y python3 python3-pip git

# Clone repository
git clone https://github.com/your-repo/h01-processor.git
cd h01-processor

# Install dependencies
pip3 install -r requirements.txt

# Setup Google Cloud credentials
gcloud auth application-default login

# Download H01 configuration
gsutil cp gs://your-bucket/h01_config.yaml .

# Start processing
python3 h01_production_pipeline.py \
    --config h01_config.yaml \
    --region large_region \
    --output gs://your-bucket/results/
```

### **4. Cost Estimation**
| Instance Type | Memory | vCPUs | Hourly Cost | Daily Cost |
|---------------|--------|-------|-------------|------------|
| n1-standard-32 | 120GB | 32 | $1.52 | $36.48 |
| n1-highmem-32 | 256GB | 32 | $2.42 | $58.08 |
| n1-highmem-64 | 512GB | 64 | $4.84 | $116.16 |

---

## ☁️ **AWS EC2 Alternative**

### **1. Setup AWS CLI**
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS
aws configure
```

### **2. Launch Memory-Optimized Instance**
```bash
# Create instance
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type r5.8xlarge \
    --key-name your-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --user-data file://setup_h01_aws.sh
```

### **3. AWS Cost Estimation**
| Instance Type | Memory | vCPUs | Hourly Cost | Daily Cost |
|---------------|--------|-------|-------------|------------|
| r5.8xlarge | 256GB | 32 | $2.016 | $48.38 |
| r5.16xlarge | 512GB | 64 | $4.032 | $96.77 |
| r5.24xlarge | 768GB | 96 | $6.048 | $145.15 |

---

## 🔧 **Distributed Processing Setup**

### **1. Multi-Instance Processing**
```python
# distributed_pipeline.py
import subprocess
import json

def distribute_processing(region_bounds, num_instances=4):
    """Distribute processing across multiple instances."""
    
    # Split region into chunks
    chunks = split_region_into_chunks(region_bounds, num_instances)
    
    # Launch instances for each chunk
    instances = []
    for i, chunk in enumerate(chunks):
        instance_name = f"h01-worker-{i}"
        
        # Launch instance
        cmd = f"""
        gcloud compute instances create {instance_name} \
            --machine-type=n1-standard-16 \
            --memory=60GB \
            --zone=us-central1-a \
            --metadata chunk='{json.dumps(chunk)}' \
            --metadata-from-file startup-script=worker_setup.sh
        """
        subprocess.run(cmd, shell=True)
        instances.append(instance_name)
    
    return instances
```

### **2. Worker Setup Script**
```bash
#!/bin/bash
# worker_setup.sh

# Get chunk data from metadata
CHUNK=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/chunk)

# Process assigned chunk
python3 h01_production_pipeline.py \
    --config h01_config.yaml \
    --region-bounds "$CHUNK" \
    --output gs://your-bucket/results/chunk_$HOSTNAME/

# Self-destruct when done
gcloud compute instances delete $HOSTNAME --zone=us-central1-a --quiet
```

---

## 📊 **Performance Comparison**

### **Processing Times (25GB Region)**
| Platform | Instance Type | Memory | Time | Cost |
|----------|---------------|--------|------|------|
| **MacBook** | M1 Pro | 16GB | ~24 hours | $0 |
| **GCP** | n1-standard-32 | 120GB | ~2 hours | $3.04 |
| **AWS** | r5.8xlarge | 256GB | ~1.5 hours | $3.02 |
| **Distributed** | 4x n1-standard-16 | 240GB | ~30 min | $1.52 |

### **Cost-Benefit Analysis**
- **MacBook**: Free but slow, limited by memory
- **Cloud Single**: Fast, moderate cost
- **Cloud Distributed**: Fastest, cost-effective for large jobs

---

## 🎯 **Recommended Deployment Strategy**

### **For Small Jobs (< 1GB)**
```bash
# Use MacBook with optimizations
python h01_production_pipeline.py \
    --config h01_config_macbook.yaml \
    --region small_region \
    --chunk-size 128
```

### **For Medium Jobs (1-10GB)**
```bash
# Use single cloud instance
gcloud compute instances create h01-medium \
    --machine-type=n1-standard-16 \
    --memory=60GB \
    --zone=us-central1-a
```

### **For Large Jobs (10GB+)**
```bash
# Use distributed processing
python distributed_pipeline.py \
    --region large_region \
    --workers 4 \
    --memory-per-worker 60GB
```

---

## 🔐 **Security Best Practices**

### **1. IAM Configuration**
```bash
# Create service account for H01 processing
gcloud iam service-accounts create h01-processor \
    --display-name="H01 Processing Service Account"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="serviceAccount:h01-processor@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="serviceAccount:h01-processor@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator"
```

### **2. Network Security**
```bash
# Create VPC with restricted access
gcloud compute networks create h01-vpc \
    --subnet-mode=auto

# Create firewall rules
gcloud compute firewall-rules create h01-allow-ssh \
    --network=h01-vpc \
    --allow=tcp:22 \
    --source-ranges=YOUR_IP/32
```

---

## 📈 **Monitoring and Logging**

### **1. Cloud Monitoring**
```python
# monitoring_setup.py
from google.cloud import monitoring_v3

def setup_monitoring():
    """Setup monitoring for H01 processing."""
    client = monitoring_v3.MetricServiceClient()
    
    # Create custom metrics
    descriptor = monitoring_v3.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/h01/processing_time"
    descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
    
    client.create_metric_descriptor(descriptor)
```

### **2. Logging Configuration**
```python
# logging_config.py
import logging
from google.cloud import logging

def setup_logging():
    """Setup structured logging."""
    client = logging.Client()
    client.setup_logging()
    
    # Create structured logger
    logger = logging.getLogger('h01_processor')
    logger.setLevel(logging.INFO)
    
    return logger
```

---

## 🚀 **Quick Start Commands**

### **1. Single Instance Deployment**
```bash
# Deploy to GCP
./deploy_gcp.sh --region large_region --memory 120GB

# Deploy to AWS
./deploy_aws.sh --instance-type r5.8xlarge --region large_region
```

### **2. Distributed Deployment**
```bash
# Deploy distributed processing
python deploy_distributed.py \
    --workers 4 \
    --memory-per-worker 60GB \
    --region large_region
```

### **3. Cost Optimization**
```bash
# Use spot instances for cost savings
gcloud compute instances create h01-spot \
    --machine-type=n1-standard-32 \
    --memory=120GB \
    --preemptible \
    --zone=us-central1-a
```

---

## 📋 **Deployment Checklist**

- [ ] **Setup Cloud Account**: GCP or AWS
- [ ] **Configure Credentials**: Service account or IAM
- [ ] **Create VPC**: Network security
- [ ] **Upload Code**: Repository or container
- [ ] **Setup Monitoring**: Logging and metrics
- [ ] **Test Small Region**: Validate setup
- [ ] **Deploy Production**: Large region processing
- [ ] **Monitor Costs**: Track spending
- [ ] **Cleanup**: Terminate instances when done

---

## 💰 **Cost Optimization Tips**

### **1. Use Spot/Preemptible Instances**
- **GCP Preemptible**: 60-80% cost savings
- **AWS Spot**: 70-90% cost savings
- **Risk**: Instances can be terminated

### **2. Right-Size Instances**
- Start with smaller instances
- Monitor resource usage
- Scale up only if needed

### **3. Batch Processing**
- Process multiple regions together
- Use sustained use discounts
- Schedule during off-peak hours

---

**🎉 Ready to process large H01 regions in the cloud! Choose the deployment option that best fits your needs and budget.** 