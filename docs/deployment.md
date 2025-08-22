# 🚀 Deployment Guide

This guide covers various deployment options for the AIS Attack Generation System, from development setups to production environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Environment](#production-environment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker (optional)
- Git

### Quick Setup

```bash
# Clone repository
git clone https://github.com/jasontimwong/ais-attack-system.git
cd ais-attack-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Verify installation
python tools/system_check.py
```

### Web Interface Setup

```bash
cd visualization/web_interface
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173`.

### Configuration

Create a local configuration file:

```bash
cp configs/default_attack_config.yaml configs/local_config.yaml
```

Edit `configs/local_config.yaml` for your local environment:

```yaml
system:
  log_level: "DEBUG"
  output_dir: "./output"

data:
  input_dir: "./data"
  cache_dir: "./cache"

performance:
  thread_count: 4
  parallel_processing: true
```

## Docker Deployment

### Single Container

Build and run the main system:

```bash
# Build image
docker build -t ais-attack-system .

# Run container
docker run -d \
  --name ais-attack \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  ais-attack-system
```

### Docker Compose (Recommended)

Deploy the complete system with all services:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale batch-worker=4

# Stop services
docker-compose down
```

#### Service Overview

```yaml
services:
  ais-attack-system:    # Main application
  web-interface:        # Web visualization
  postgres:            # Database
  redis:               # Cache and job queue
  batch-worker:        # Background processing
  prometheus:          # Metrics collection
  grafana:             # Monitoring dashboard
  elasticsearch:       # Log aggregation
  logstash:           # Log processing
  kibana:             # Log visualization
```

### Development Environment

For development with live code reloading:

```bash
# Start development environment
docker-compose --profile development up -d

# Access development container
docker-compose exec dev-environment bash

# Run tests inside container
docker-compose run test-runner
```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Build and push Docker image**:

```bash
# Build for production
docker build -t ais-attack-system:latest .

# Tag for ECR
docker tag ais-attack-system:latest \
  123456789012.dkr.ecr.us-west-2.amazonaws.com/ais-attack-system:latest

# Push to ECR
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/ais-attack-system:latest
```

2. **Create ECS task definition**:

```json
{
  "family": "ais-attack-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ais-attack-system",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/ais-attack-system:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ais-attack-system",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Deploy with Terraform**:

```hcl
# main.tf
resource "aws_ecs_cluster" "ais_attack_cluster" {
  name = "ais-attack-system"
}

resource "aws_ecs_service" "ais_attack_service" {
  name            = "ais-attack-service"
  cluster         = aws_ecs_cluster.ais_attack_cluster.id
  task_definition = aws_ecs_task_definition.ais_attack_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.ais_attack_sg.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ais_attack_tg.arn
    container_name   = "ais-attack-system"
    container_port   = 8000
  }
}
```

#### Using Lambda for Batch Processing

```python
# lambda_handler.py
import json
import boto3
from core.attack_orchestrator import AttackOrchestrator

def lambda_handler(event, context):
    """
    AWS Lambda handler for batch attack processing
    """
    # Initialize components
    orchestrator = AttackOrchestrator()
    
    # Process attack request
    attack_params = json.loads(event['body'])
    results = orchestrator.execute_attack(attack_params)
    
    # Store results in S3
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket='ais-attack-results',
        Key=f"results/{results['attack_id']}.json",
        Body=json.dumps(results)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'attack_id': results['attack_id'],
            'success': results['success']
        })
    }
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/ais-attack-system

gcloud run deploy ais-attack-system \
  --image gcr.io/PROJECT_ID/ais-attack-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80
```

#### Using GKE (Google Kubernetes Engine)

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ais-attack-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ais-attack-system
  template:
    metadata:
      labels:
        app: ais-attack-system
    spec:
      containers:
      - name: ais-attack-system
        image: gcr.io/PROJECT_ID/ais-attack-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ais-attack-service
spec:
  selector:
    app: ais-attack-system
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Microsoft Azure

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name ais-attack-rg --location eastus

# Deploy container
az container create \
  --resource-group ais-attack-rg \
  --name ais-attack-system \
  --image ais-attack-system:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables LOG_LEVEL=INFO
```

## Production Environment

### High Availability Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ais-attack-system:
    image: ais-attack-system:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:pass@postgres:5432/aisattack
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "python", "tools/system_check.py", "--quiet"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - ais-attack-system

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=aisattack
      - POSTGRES_USER=aisuser
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - postgres_password
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    deploy:
      replicas: 1

volumes:
  postgres_data:
  redis_data:

secrets:
  postgres_password:
    external: true
```

### Load Balancer Configuration

```nginx
# nginx.conf
upstream ais_attack_backend {
    least_conn;
    server ais-attack-system_1:8000;
    server ais-attack-system_2:8000;
    server ais-attack-system_3:8000;
}

server {
    listen 80;
    server_name ais-attack.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ais-attack.example.com;

    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    location / {
        proxy_pass http://ais_attack_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

### Database Migration

```python
# migrations/001_initial_schema.py
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import String, Integer, Float, DateTime, JSON, Boolean

def upgrade(engine):
    metadata = MetaData()
    
    # Attack results table
    attack_results = Table('attack_results', metadata,
        Column('id', String, primary_key=True),
        Column('attack_type', String, nullable=False),
        Column('target_mmsi', String, nullable=False),
        Column('ghost_mmsi', String, nullable=False),
        Column('execution_time', Float),
        Column('success', Boolean),
        Column('results', JSON),
        Column('created_at', DateTime),
        Column('updated_at', DateTime)
    )
    
    # Scenarios table
    scenarios = Table('scenarios', metadata,
        Column('id', String, primary_key=True),
        Column('name', String, nullable=False),
        Column('config', JSON),
        Column('status', String),
        Column('created_at', DateTime)
    )
    
    metadata.create_all(engine)

def downgrade(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(engine)
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ais-attack-system'
    static_configs:
      - targets: ['ais-attack-system:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AIS Attack System Monitoring",
    "panels": [
      {
        "title": "Attack Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ais_attacks_successful_total[5m]) / rate(ais_attacks_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "Processing Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ais_messages_processed_total[5m])"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes"
          },
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

```yaml
# monitoring/logstash/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "ais-attack-system" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{WORD:logger} - %{LOGLEVEL:level} - %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ais-attack-logs-%{+YYYY.MM.dd}"
  }
}
```

## Security Considerations

### Network Security

```yaml
# docker-compose.security.yml
version: '3.8'

services:
  ais-attack-system:
    networks:
      - internal
      - web
    environment:
      - ALLOWED_HOSTS=ais-attack.example.com
      - CORS_ORIGINS=https://ais-attack.example.com

  postgres:
    networks:
      - internal
    environment:
      - POSTGRES_SSL_MODE=require

  redis:
    networks:
      - internal
    command: redis-server --requirepass ${REDIS_PASSWORD}

networks:
  web:
    external: true
  internal:
    internal: true
```

### SSL/TLS Configuration

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=ais-attack.example.com"
```

### Environment Variables

```bash
# .env.production
DATABASE_URL=postgresql://user:${DB_PASSWORD}@postgres:5432/aisattack
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
LOG_LEVEL=INFO
DEBUG=false
```

### Access Control

```python
# security/auth.py
from functools import wraps
from flask import request, jsonify
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits
docker run -m 4g ais-attack-system

# Monitor memory usage
docker-compose exec ais-attack-system python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### 2. Performance Issues

```bash
# Check CPU usage
top -p $(docker inspect --format '{{.State.Pid}}' ais-attack-system)

# Profile application
docker-compose exec ais-attack-system python -m cProfile -o profile.stats -m core.attack_orchestrator

# Analyze profile
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(10)
"
```

#### 3. Database Connection Issues

```bash
# Check database connectivity
docker-compose exec ais-attack-system python -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://user:pass@postgres:5432/aisattack')
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database logs
docker-compose logs postgres
```

#### 4. Web Interface Issues

```bash
# Check web interface logs
docker-compose logs web-interface

# Rebuild web interface
docker-compose exec web-interface npm run build

# Check network connectivity
curl http://localhost:5173/health
```

### Health Checks

```python
# tools/health_check.py
import requests
import sys
import psycopg2
import redis

def check_api_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_database_health():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        conn.close()
        return True
    except:
        return False

def check_redis_health():
    try:
        r = redis.from_url(os.environ['REDIS_URL'])
        r.ping()
        return True
    except:
        return False

if __name__ == '__main__':
    checks = [
        ('API', check_api_health),
        ('Database', check_database_health),
        ('Redis', check_redis_health)
    ]
    
    all_healthy = True
    for name, check_func in checks:
        healthy = check_func()
        status = '✅' if healthy else '❌'
        print(f'{status} {name}: {"Healthy" if healthy else "Unhealthy"}')
        all_healthy &= healthy
    
    sys.exit(0 if all_healthy else 1)
```

### Log Analysis

```bash
# View application logs
docker-compose logs -f ais-attack-system

# Search for errors
docker-compose logs ais-attack-system | grep ERROR

# Monitor real-time logs
tail -f logs/attack_generation.log | grep -E "(ERROR|WARNING)"

# Analyze log patterns
awk '/ERROR/ {print $1, $2, $NF}' logs/attack_generation.log | sort | uniq -c
```

---

This deployment guide provides comprehensive coverage for deploying the AIS Attack Generation System in various environments. Choose the deployment method that best fits your requirements and infrastructure.
