# FLoV2T Installation and Setup Guide

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 10GB+ disk space

## Step 1: Create Virtual Environment

```bash
cd "/mnt/c/Users/christopherli/OneDrive - TrendMicro/Apey/Masteral/Papers/EdgeFedIDS/benchmark_suite/implementation/flov2t"

# Create virtual environment
python -m venv venv

# Activate (Windows WSL)
source venv/bin/activate

# Or activate (Windows CMD)
# venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import ViTForImageClassification; print('Transformers OK')"
python -c "from scapy.all import rdpcap; print('Scapy OK')"
```

## Step 4: Download CICIDS2017 Dataset

### Option A: Manual Download

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download PCAP files
3. Place in: `../../datasets/CICIDS2017/raw/`

### Option B: Use Existing Dataset

If you already have processed CICIDS2017 data:
```bash
# Update config.yaml to point to your data
vim config/config.yaml
# Change dataset.root_dir to your data location
```

## Step 5: Directory Structure

Ensure this structure:

```
EdgeFedIDS/
├── benchmark_suite/
│   └── implementation/
│       └── flov2t/           # This project
│           ├── config/
│           ├── data/
│           ├── models/
│           ├── federated/
│           ├── utils/
│           └── train.py
└── datasets/
    └── CICIDS2017/
        ├── raw/              # Original PCAP files
        │   ├── Friday-WorkingHours-Afternoon-DDos.pcap
        │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap
        │   └── ...
        └── processed/         # Processed images (created by preprocessing)
            ├── train/
            │   ├── Botnet/
            │   ├── DoS-Slowloris/
            │   └── ...
            └── test/
```

## Step 6: Preprocess Data

```bash
# This will take several hours depending on dataset size
python preprocess_cicids.py \
    --input ../../datasets/CICIDS2017/raw \
    --output ../../datasets/CICIDS2017/processed \
    --attack-types Botnet DoS-Slowloris DoS-Goldeneye DoS-Hulk SSH-BruteForce Web-SQL Web-XSS Web-Bruteforce
```

## Step 7: Test Components

```bash
# Test Packet2Patch
python -m data.packet2patch

# Test LoRA
python -m models.lora

# Test RTFE
python -m models.rtfe
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config/config.yaml`
- Use CPU: set `hardware.device: "cpu"`

### Scapy Import Error
```bash
# On Windows WSL, install npcap
sudo apt-get install tcpdump libpcap-dev
```

### Transformers Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
```

### Permission Errors on PCAP
```bash
# Run with sudo or adjust permissions
chmod +r ../../datasets/CICIDS2017/raw/*.pcap
```

## Quick Start Training

Once setup is complete:

```bash
# IID, 3 clients
python train.py --config config/config.yaml --num_clients 3 --distribution iid --rounds 18

# Non-IID, 5 clients  
python train.py --config config/config.yaml --num_clients 5 --distribution non_iid --rounds 18
```

## Expected Resource Usage

- **Preprocessing**: 2-4 hours, 8GB RAM
- **Training (per round)**: 5-10 minutes, 4GB GPU memory
- **Total training (18 rounds)**: 2-3 hours
- **Disk space**: ~5GB for processed data

## Next Steps

After installation:
1. Read `README.md` for usage examples
2. Check `config/config.yaml` for configuration options
3. Run preprocessing on a small subset first
4. Start training with IID scenario
5. Evaluate on test set

## Support

For issues:
1. Check logs in `./logs/`
2. Verify configuration in `config/config.yaml`
3. Test individual components
4. Check GPU memory with `nvidia-smi`
