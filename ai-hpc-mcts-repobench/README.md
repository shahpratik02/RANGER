# RepoBench Evaluation

This repository contains code to run RepoBench evaluation using graph-based approaches with Neo4j database integration.

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

## Neo4j Database Setup for Amazon EC2 Linux 2023

### Installation (Amazon Linux/RHEL/CentOS)

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

Once the setup is complete, you can run the following evaluation scripts:

### 1. Graph Evaluation

```bash
python -m scripts.RepoBench_end_to_end --retriever graph
```

### 2. File_level Evaluation (Baseline)

```bash
python -m scripts.RepoBench_end_to_end --retriever file_level
```

### 3. Semantic Evaluation (Very poor so no use) 

```bash
python -m scripts.RepoBench_end_to_end --retriever semantic
```

## Configuration

All configurations are located in:
```
/home/ec2-user/to_push/config.yaml
```

## Neo4j Dump Instructions

To store the graphs we create on Hugging Face, set the `neo4j_dump` flag to `True` in `config.yaml`.  
Then, follow these steps:

1. **Log in to Hugging Face**
   ```bash
   # Activate your virtual environment first
   huggingface-cli login
   ```

2. **Create the directory for Neo4j dumps**
   ```bash
   mkdir -p /home/ec2-user/neo4j_dumps
   ```

3. **Configure passwordless sudo for specific commands**
   ```bash
   sudo visudo
   ```
   Add the following lines at the end of the file, replacing `ec2-user` with your username:

   ```
   # Allows ec2-user to stop/start Neo4j and manage file ownership without a password
   ec2-user ALL=(ALL) NOPASSWD: /bin/systemctl stop neo4j, /bin/systemctl start neo4j, /bin/chown, /bin/mv

   # Allows ec2-user to run neo4j-admin as the neo4j user without a password
   ec2-user ALL=(neo4j) NOPASSWD: /usr/bin/neo4j-admin


5. **Change the group ownership of the directory to the 'neo4j' user's group**
    ```bash
    sudo chown ec2-user:neo4j /home/ec2-user/neo4j_dumps
    ```

3. **Set permissions to allow the owner (ec2-user) and the group (neo4j) to write to it**
   ```bash
    sudo chmod 775 /home/ec2-user/neo4j_dump
   ```
