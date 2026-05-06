# Homework: Distributed GPU Training on Nebius Cloud with SkyPilot
**Estimated time:** 3–5 hours  
**Platform:** Nebius Cloud (mk8s + Container registry + SkyPilot API)

---

## Overview

In this homework you will set up a distributed training MLOps pipeline on Nebius Cloud:
provision a managed Kubernetes cluster with GPU nodes, deploy a SkyPilot API server,
containerize a training workload, and launch a distributed training job using PyTorch DDP.

> Use as a reference: (https://gitlab.com/jadnov/nebius-academy-ddp)

### Deliverables

At the end of the homework, submit:

1. ✅ mk8s node-group configuration 
2. ✅ Your `Dockerfile` and `train.py`
3. ✅ Your SkyPilot job YAML (`train_job.yaml`)
4. ✅ Full training log (from `sky logs`) that includes the NCCL initialization section

---

## Prerequisites: Install the Nebius AI Cloud CLI 

```
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
```

which manages all Nebius AI Cloud resources: https://docs.nebius.com/cli/install

## Task 1 — Create an mk8s Cluster on Nebius Cloud

### Goal

Provision a Nebius Managed Kubernetes (mk8s) cluster with a single nodegroup using a **1-GPU node preset**.

### Steps

1. Log in to the [Nebius Cloud Console](https://console.nebius.com).
2. Navigate to **Managed Kubernetes → Clusters → Create cluster**.
3. Configure the cluster:
   - **Name:** choose a name for your cluster
   - **Kubernetes version:** latest stable
   - **Network:** default VPC
4. Add a **Node Group**:
   - **Name:** name for a node-group
   - **Node preset:** `gpu-h100-b-1gpu` *(Alternatives: `gpu-l40s-1gpu` for L40S)*
   - **Nodes:** `2`
   - **Disk:** 100 GB SSD
5. Create `Service Account` and add it to `Viewers` group (to allow access to container registry from nodes)
6. Click **Create** and wait for the cluster to reach **Running** status (~5 min).
7. Download the kubeconfig:

```bash
nebius mk8s cluster get-credentials \
  --id <YOUR_CLUSTER_ID> \
  --external \
  --kubeconfig ~/.kube/config
```

7. Verify connectivity:

```bash
kubectl get nodes
```

### Expected Output

```
NAME                                 STATUS   ROLES    AGE     VERSION
computeinstance-e00gt41yn2fpng6qzm   Ready    <none>   5m10s   v1.33.7
computeinstance-e00r2v341f0f616jjq   Ready    <none>   4m59s   v1.33.7
```

As delivarble from this step prepare node-group config in following format:
```
nebius mk8s node-group get --id <node-group-id> --format json | jq '{metadata, spec}'
```

---

## Task 2 — Deploy SkyPilot API Server & Connect a Client

### Goal

Deploy the SkyPilot API server as Nebius managed service,
then configure the local `sky` CLI to talk to it.

### Steps

#### 2a. Deploy the Managed SkyPilot API Server

1. In the Nebius AI Cloud console, go to AI Services → SkyPilot.
2. Enter a name for the application or keep the default one.
3. Select a Platform and a Preset (4 vCPUs and 16G RAM) for the API server VM.
4. Click Deploy application and wait for the Public endpoint availability (~5 min).

#### 2b. Install SkyPilot locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --with pip "skypilot[nebius]"
sky api login -e "https://<your_api_server_public_endpoint>"
sky check nebius
```

Verify the connection:

```bash
sky api info
sky check kubernetes
```

### Expected Output

```
$ sky check kubernetes
🎉 Enabled infra 🎉
  Kubernetes [compute]
    Allowed contexts:
    └── <your_mk8s_cluster_name>
```

---

## Task 3 — Build a Docker Container for Training

### Goal

Create a Docker image with all dependencies needed to train model using PyTorch DDP.

### Dockerfile

Create a file named `Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    wandb \
    scipy

CMD ["bash"]
```

### Training Script

Create a file named `train.py`. This script train a small causal LM with DDP:

> **Alternative models for faster testing:** 
> `"facebook/opt-1.3b"` or `"tiiuae/falcon-7b"`
> both are publicly available and work identically with the script.

### Create Docker Registry on Nebius cloud

1. Navigate to **Storage → Container registry → Create registry**.
2. Run the command that sets up the CLI as a Docker credential helper for Nebius registries:
```
nebius registry configure-helper
```

### Build Docker container from Dockerfile & Push to registry

```bash
# Build from Dockerfile in the local directory
docker build -t <registry>/nebius-trainer:v1 .
# Push to Nebius Container Registry (or any registry your cluster can pull from)
docker push <registry>/nebius-trainer:v1
```

> **Hint:** Nebius Container Registry endpoint looks like:
> `cr.<region>.nebius.cloud/<registry-id>/`

---

## Task 4 — Write the SkyPilot Job YAML with PyTorch DDP

### Goal

Write a SkyPilot task YAML that launches the training container on your Kubernetes cluster
using `torchrun` for DDP.

### `train_job.yaml`

```yaml
name: nebius-ddp-training

workdir: .

resources:
  infra: k8s/<mk8s-cluster-name>
  accelerators: "H100:1"
  memory: "60+"
  image_id: docker:cr.<region>.nebius.cloud/<registry-id>/nebius-trainer:v1

num_nodes: 2

envs:
  MODEL_ID: "facebook/opt-2.7b"
  TRAIN_SCRIPT: "train.py"
  BLOCK_SIZE: "512"
  PER_DEVICE_TRAIN_BATCH_SIZE: "4"
  PER_DEVICE_EVAL_BATCH_SIZE: "4"
  GRADIENT_ACCUMULATION_STEPS: "1"
  DATALOADER_NUM_WORKERS: "8"
  TOKENIZERS_PARALLELISM: "false"
  NCCL_DEBUG: INFO
  NCCL_DEBUG_SUBSYS: INIT,NET

setup: |
  echo "Setup complete"
  nvidia-smi || true

run: |
  set -euxo pipefail

  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n 1)
  MASTER_PORT=29500

  echo "SKYPILOT_NODE_RANK=${SKYPILOT_NODE_RANK}"
  echo "SKYPILOT_NUM_NODES=${SKYPILOT_NUM_NODES}"
  echo "SKYPILOT_NUM_GPUS_PER_NODE=${SKYPILOT_NUM_GPUS_PER_NODE}"
  echo "SKYPILOT_NODE_IPS:"
  echo "$SKYPILOT_NODE_IPS"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "PWD=$(pwd)"
  test -f "${TRAIN_SCRIPT}"

  torchrun \
    --nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE} \
    --nnodes=${SKYPILOT_NUM_NODES} \
    --node_rank=${SKYPILOT_NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${TRAIN_SCRIPT}"
```

> **Key SkyPilot environment variables used:**
>
> | Variable | Meaning |
> |---|---|
> | `SKYPILOT_NUM_GPUS_PER_NODE` | Number of GPUs on this node |
> | `SKYPILOT_NUM_NODES` | Total nodes in the job |
> | `SKYPILOT_NODE_RANK` | Rank of this node (0 = head) |
> | `SKYPILOT_INTERNAL_HEAD_IP` | IP of the head node for rendezvous |

> **Hint — NCCL debug logs:** Setting `NCCL_DEBUG=INFO` is essential for this homework —
> it makes NCCL print initialisation details
> that you need to include in your submission.

---

## Task 5 — Run Training for 500 Steps & Collect Results

### Goal

Submit the job, wait for it to finish, and capture training logs with NCCL init section

### 5a. Submit the job

```bash
sky launch -c ddp-run train_job.yaml
```

Watch live logs:

```bash
sky logs ddp-run
```

### 5b. Save the full log

```bash
sky logs ddp-run > training_log.txt
```

Verify the log contains NCCL init lines — look for patterns like:

```
[0] NCCL INFO Bootstrap : Using eth0:10.x.x.x<0>
[0] NCCL INFO NET/Plugin : No plugin found, using internal net plugin
```

### 5c. Clean up

```bash
sky down ddp-run
```

---

## Submission Checklist

| # | Item | File name |
|---|------|-----------|
| 1 | mk8s node-group configuration  | `mk8s-ng-config.json` |
| 2 | Dockerfile | `Dockerfile` |
| 3 | Training script | `train.py` |
| 4 | SkyPilot job YAML | `train_job.yaml` |
| 5 | Full training log (must include NCCL init) | `training_log.txt` |

---

## Grading Criteria

| Task | Points |
|------|--------|
| mk8s cluster running with correct setup of GPU nodes | 15 |
| Dockerfile builds successfully | 15 |
| `train.py` runs correctly | 20 |
| `train_job.yaml` correct (torchrun + SkyPilot vars) | 20 |
| Training log contains NCCL init section | 20 |
| **Total** | **100** |

---

## Useful Resources

- [Nebius Cloud docs — mk8s](https://docs.nebius.com/mk8s/)
- [Nebius Cloud docs — Container registry](https://docs.nebius.com/container-registry/quickstart)
- [Nebius Cloud docs — SkyPilot](https://docs.nebius.com/3p-integrations/skypilot)
- [SkyPilot documentation](https://skypilot.readthedocs.io/)
- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

---