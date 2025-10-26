# FileIntel Production Deployment Guide

Production deployment guide covering security, scaling, and advanced configurations. For quick deployment, see [README](README.md#deployment).

## Table of Contents

1. [Deployment Methods](#deployment-methods)
2. [SSL/TLS Setup](#ssltls-setup)
3. [Security Hardening](#security-hardening)
4. [Scaling](#scaling)
5. [Monitoring](#monitoring)
6. [Backup Strategy](#backup-strategy)

---

## Deployment Methods

### Git-Based (Recommended)

```bash
git clone https://github.com/yourusername/fileintel.git /opt/fileintel
cd /opt/fileintel
git checkout v1.0.0  # Tag specific version
cp .env.example .env
nano .env
mkdir -p logs uploads input output graphrag_indices
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

**Pros:** Simple updates via `git pull`, version control
**Cons:** Source visible on server, build time on deployment

---

### Docker Registry (CI/CD)

Push to registry:
```bash
docker tag fileintel-api:latest yourusername/fileintel-api:v1.0
docker push yourusername/fileintel-api:v1.0
```

Update `docker-compose.prod.yml`:
```yaml
services:
  api:
    image: yourusername/fileintel-api:v1.0
  celery-worker:
    image: yourusername/fileintel-worker:v1.0
  flower:
    image: yourusername/fileintel-flower:v1.0
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

**Pros:** Fast deployments, version tagging, easy rollbacks
**Cons:** Registry setup required

---

### Image Transfer (Air-gapped)

```bash
# Development machine
docker-compose build
docker save fileintel-api:latest fileintel-celery-worker:latest fileintel-flower:latest -o fileintel.tar
gzip fileintel.tar

# Production server
gunzip fileintel.tar.gz
docker load -i fileintel.tar
docker-compose -f docker-compose.prod.yml up -d
```

**Pros:** Works offline, pre-built images
**Cons:** Large file transfers (~5-7GB), manual updates

---

## SSL/TLS Setup

### Nginx Reverse Proxy + Let's Encrypt

Install:
```bash
apt install nginx certbot python3-certbot-nginx
```

Configure `/etc/nginx/sites-available/fileintel`:
```nginx
server {
    listen 80;
    server_name fileintel.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name fileintel.example.com;

    ssl_certificate /etc/letsencrypt/live/fileintel.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fileintel.example.com/privkey.pem;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable:
```bash
ln -s /etc/nginx/sites-available/fileintel /etc/nginx/sites-enabled/
certbot --nginx -d fileintel.example.com
systemctl restart nginx
```

For Traefik or other reverse proxies, see [Traefik docs](https://doc.traefik.io/traefik/).

---

## Security Hardening

### Network Security

```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 5432/tcp  # PostgreSQL
ufw deny 6379/tcp  # Redis
ufw deny 5555/tcp  # Flower (or whitelist specific IPs)
ufw enable
```

### Docker Secrets

Replace environment variables with Docker secrets:

```yaml
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  openai_api_key:
    file: ./secrets/openai_api_key.txt

services:
  api:
    secrets:
      - postgres_password
      - openai_api_key
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
```

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Database Permissions

```sql
-- Revoke public access
REVOKE ALL ON DATABASE fileintel FROM PUBLIC;

-- Grant minimal permissions
GRANT CONNECT ON DATABASE fileintel TO fileintel_user;
GRANT USAGE ON SCHEMA public TO fileintel_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fileintel_user;
```

---

## Scaling

### Horizontal Scaling

**Multiple Celery workers:**
```yaml
services:
  celery-worker:
    deploy:
      replicas: 3
```

Or manually:
```bash
docker-compose -f docker-compose.prod.yml up -d --scale celery-worker=5
```

**Load-balanced API:**
```yaml
services:
  api-1:
    image: fileintel-api:latest
    # ... config

  api-2:
    image: fileintel-api:latest
    # ... config

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8000:80"
```

`nginx-lb.conf`:
```nginx
upstream api_servers {
    least_conn;
    server api-1:8000;
    server api-2:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_servers;
    }
}
```

### Performance Tuning

**Production config (`config/default.yaml`):**
```yaml
# Database - higher connection pool
storage:
  pool_size: 50
  max_overflow: 50

# Celery - more concurrency, longer timeouts
celery:
  worker_concurrency: 8
  task_soft_time_limit: 7200
  task_time_limit: 7200
  worker_max_tasks_per_child: 1000

# RAG - async optimizations
rag:
  async_processing:
    enabled: true
    batch_size: 10
    max_concurrent_requests: 25
```

**PostgreSQL tuning:**
For large deployments, increase shared buffers and work memory. See [PostgreSQL tuning guide](https://wiki.postgresql.org/wiki/Tuning_Your_PostgreSQL_Server).

**Redis optimization:**
```yaml
services:
  redis:
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --save ""
```

---

## Monitoring

### Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:5555/api/workers
docker-compose exec postgres pg_isready -U fileintel_user
docker-compose exec redis redis-cli ping
```

### Log Aggregation

Use Loki + Grafana for centralized logging:

```yaml
services:
  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - ./logs:/var/log/fileintel
      - ./promtail-config.yaml:/etc/promtail/config.yml

  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=<password>
```

### Metrics with Prometheus

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
```

---

## Backup Strategy

### Automated Backups

The production compose includes a daily backup service. Verify it's running:
```bash
docker-compose logs backup
```

### Manual Backup

**Full backup:**
```bash
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Database
docker-compose exec -T postgres pg_dump -U fileintel_user fileintel | gzip > "$BACKUP_DIR/database.sql.gz"

# Configuration
tar czf "$BACKUP_DIR/config.tar.gz" config/ prompts/ .env docker-compose.prod.yml

# Volumes
docker run --rm -v fileintel_postgres_data:/data -v "$BACKUP_DIR":/backup \
  alpine tar czf /backup/postgres_data.tar.gz -C /data .
```

### Restore

```bash
gunzip < backup.sql.gz | docker-compose exec -T postgres psql -U fileintel_user fileintel
```

### Offsite Backups

Upload to S3/B2:
```bash
aws s3 sync /backups/ s3://your-bucket/fileintel-backups/
```

**Best practices:**
- Test restores monthly
- Keep 30 days of daily backups
- Keep 12 months of monthly backups
- Store backups in different region/provider

---

## System Requirements

**Minimum:**
- 4 CPU cores
- 16GB RAM
- 100GB disk space
- Docker 20.10+
- Docker Compose 2.0+

**Recommended:**
- 8+ CPU cores
- 32GB+ RAM
- 500GB SSD
- GPU for local LLM inference (optional)

---

## Common Issues

**Services fail to start:**
```bash
docker-compose -f docker-compose.prod.yml logs
docker-compose -f docker-compose.prod.yml ps
```

**Out of memory:**
```bash
docker stats
# Increase limits in docker-compose.prod.yml
# Reduce worker_concurrency in config/default.yaml
```

**Database connection errors:**
```bash
docker-compose exec postgres psql -U fileintel_user -d fileintel -c "SELECT 1;"
docker-compose exec api env | grep DB_
```

---

## Production Checklist

Before going live:

- [ ] SSL/TLS configured and tested
- [ ] Firewall rules applied
- [ ] Strong passwords set (20+ chars)
- [ ] Docker secrets configured
- [ ] Backup strategy tested
- [ ] Monitoring/alerting set up
- [ ] Resource limits configured
- [ ] Flower dashboard restricted
- [ ] API authentication enabled
- [ ] Offsite backups configured

---

For basic deployment, see [README.md](README.md#deployment).
