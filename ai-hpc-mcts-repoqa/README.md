# RepoQA Evaluation

This repository contains code to run RepoQA evaluation using graph-based approaches with Neo4j database integration.

## Prerequisites

- Python 3.9
- Neo4j database
- Required data files

## Initial Setup


### 2. Python Environment Setup

Create the Python environment using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Note:** Ensure you're using Python 3.9 for compatibility.

## Neo4j Database Setup

### Installation (Amazon Linux/RHEL/CentOS) - For easiest setup use the AMI Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Amazon Linux 2023) 20250726 and Python 3.9.23

```bash
# Update system
sudo dnf update -y

# Install Java 17
sudo dnf install -y java-17-amazon-corretto-devel

# Import Neo4j GPG key
sudo rpm --import https://debian.neo4j.com/neotechnology.gpg.key

# Add Neo4j repository
cat <<EOF | sudo tee /etc/yum.repos.d/neo4j.repo
[neo4j]
name=Neo4j RPM Repository
baseurl=https://yum.neo4j.com/stable/5
enabled=1
gpgcheck=1
EOF

# Install Neo4j
sudo dnf install -y neo4j-5.26.0
```

### Plugin Installation

```bash
# Create plugins directory
sudo mkdir -p /var/lib/neo4j/plugins
sudo chown -R neo4j:neo4j /var/lib/neo4j

# Download APOC plugin
sudo wget -O /var/lib/neo4j/plugins/apoc-5.26.0-core.jar \
  https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.26.0/apoc-5.26.0-core.jar

# Download Graph Data Science plugin
sudo wget -O /var/lib/neo4j/plugins/neo4j-graph-data-science-2.13.4.jar \
  https://github.com/neo4j/graph-data-science/releases/download/2.13.4/neo4j-graph-data-science-2.13.4.jar

# Set permissions
sudo chown neo4j:neo4j /var/lib/neo4j/plugins/*.jar
sudo chmod 644 /var/lib/neo4j/plugins/*.jar
```

### Configuration

```bash
# Configure Neo4j to enable plugins
sudo tee -a /etc/neo4j/neo4j.conf <<EOF
# Enable APOC procedures
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*
EOF
```

### Service Management

```bash
# Enable and start Neo4j service
sudo systemctl enable neo4j
sudo systemctl start neo4j

# Check service status
sudo systemctl status neo4j
```

### Initial Setup

```bash
# Connect to Neo4j (you'll be prompted to create a new password)
cypher-shell -u neo4j -p neo4j
```

## Data Loading

### Download Database Dump

Download the Neo4j database dump from:
[https://huggingface.co/datasets/Nutanix/RepoQA-neo4j/blob/main/neo4j.dump](https://huggingface.co/datasets/Nutanix/RepoQA-neo4j/blob/main/neo4j.dump)

### Load Database

```bash
# Stop Neo4j service
sudo systemctl stop neo4j

# Copy dump file to Neo4j directory
sudo cp /path/to/your/neo4j.dump /var/lib/neo4j/
sudo chown neo4j:neo4j /var/lib/neo4j/neo4j.dump

# Load the database
sudo -u neo4j neo4j-admin database load neo4j --from-path=/var/lib/neo4j --overwrite-destination=true 

# Start Neo4j service
sudo systemctl start neo4j
```

## Evaluation Scripts

Once the setup is complete, you can run the following evaluation scripts:

### 1. MCTS Evaluation

```bash
python -m scripts.RepoQA_eval
```

### 2. Text Embedding Baseline

```bash
python -m scripts.RepoQA_eval --text_semantic
```

### 3. Code Embedding Baseline

```bash
python -m scripts.RepoQA_eval --code_semantic
```

## Configuration

All configurations are located in:
```
config.yaml
```

## Creating Additional Graphs

To create graphs for more repositories, use the data download script:
```
src/data/repoqa_data_creation.py
```
