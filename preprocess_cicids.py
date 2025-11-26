import argparse
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import logging
from scapy.all import rdpcap, wrpcap
from data.packet2patch import Packet2PatchTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CICIDS2017Preprocessor:
    """
    Preprocess CICIDS2017 using CSV labels to extract specific flows from PCAP files
    and convert them to images using Packet2Patch transformation
    
    Based on FLoV2T paper Table 1 - extracts only 8 attack categories with small sample counts
    """
    
    def __init__(self, 
                 pcap_dir: str,
                 csv_dir: str,
                 output_dir: str,
                 attack_types: dict = None,
                 train_ratio: float = 0.8):
        """
        Args:
            pcap_dir: Directory containing raw PCAP files
            csv_dir: Directory containing CSV label files
            output_dir: Directory to save processed images
            attack_types: Dict mapping our attack names to CSV label patterns
            train_ratio: Ratio of training vs test split
        """
        self.pcap_dir = Path(pcap_dir)
        self.csv_dir = Path(csv_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        if attack_types is None:
            self.attack_mapping = {
                'Botnet': ['Bot'],
                'DoS-Slowloris': ['DoS slowloris', 'DoS Slowloris'],
                'DoS-Goldeneye': ['DoS GoldenEye', 'DoS Goldeneye'],
                'DoS-Hulk': ['DoS Hulk', 'DoS hulk'],
                'SSH-BruteForce': ['SSH-Patator'],
                'Web-SQL': ['Web Attack � Sql Injection', 'Web Attack - Sql Injection'],
                'Web-XSS': ['Web Attack � XSS', 'Web Attack - XSS'],
                'Web-Bruteforce': ['Web Attack � Brute Force', 'Web Attack - Brute Force']
            }
        else:
            self.attack_mapping = attack_types
        
        self.transformer = Packet2PatchTransformer(
            max_packets=196,
            patch_size=16,
            net_header_bytes=20,
            trans_header_bytes=20,
            payload_bytes=216
        )
        
        self.train_dir = self.output_dir / 'train'
        self.test_dir = self.output_dir / 'test'
        
        for attack_type in self.attack_mapping.keys():
            (self.train_dir / attack_type).mkdir(parents=True, exist_ok=True)
            (self.test_dir / attack_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessor initialized")
        logger.info(f"PCAP directory: {self.pcap_dir}")
        logger.info(f"CSV directory: {self.csv_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Attack types: {list(self.attack_mapping.keys())}")
    
    def find_csv_pcap_pairs(self):
        """
        Find matching CSV and PCAP file pairs
        
        Returns:
            List of tuples (csv_path, pcap_path)
        """
        csv_files = list(self.csv_dir.glob('*.csv'))
        pairs = []
        
        for csv_file in csv_files:
            base_name = csv_file.stem.replace('_ISCX', '').replace('.pcap', '')
            
            pcap_candidates = [
                self.pcap_dir / f"{base_name}.pcap",
                self.pcap_dir / f"{base_name}.pcapng",
            ]
            
            for pcap_file in pcap_candidates:
                if pcap_file.exists():
                    pairs.append((csv_file, pcap_file))
                    logger.info(f"Found pair: {csv_file.name} <-> {pcap_file.name}")
                    break
        
        return pairs
    
    def load_csv_labels(self, csv_file: Path):
        """
        Load CSV file and extract flow labels
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            DataFrame with flow information and labels
        """
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
        except:
            try:
                df = pd.read_csv(csv_file, encoding='latin1', low_memory=False)
            except Exception as e:
                logger.error(f"Error reading CSV {csv_file}: {e}")
                return None
        
        df.columns = df.columns.str.strip()
        
        if 'Label' not in df.columns and ' Label' not in df.columns:
            logger.error(f"No Label column found in {csv_file}")
            return None
        
        label_col = 'Label' if 'Label' in df.columns else ' Label'
        df['Label'] = df[label_col].str.strip()
        
        attack_flows = df[df['Label'] != 'BENIGN'].copy()
        
        logger.info(f"Loaded {len(attack_flows)} attack flows from {csv_file.name}")
        
        return attack_flows
    
    def map_csv_label_to_attack_type(self, csv_label: str):
        """
        Map CSV label to our attack type
        
        Args:
            csv_label: Label from CSV file
        
        Returns:
            Our attack type name or None
        """
        for attack_type, patterns in self.attack_mapping.items():
            for pattern in patterns:
                if pattern.lower() in csv_label.lower():
                    return attack_type
        return None
    
    def extract_flows_from_pcap(self, pcap_file: Path, flow_labels: pd.DataFrame):
        """
        Extract specific flows from PCAP based on CSV labels
        
        Args:
            pcap_file: Path to PCAP file
            flow_labels: DataFrame with labeled flows
        
        Returns:
            Dictionary mapping attack_type -> list of flow packets
        """
        logger.info(f"Reading PCAP: {pcap_file.name}")
        
        try:
            packets = rdpcap(str(pcap_file))
        except Exception as e:
            logger.error(f"Error reading PCAP {pcap_file}: {e}")
            return {}
        
        logger.info(f"Total packets in PCAP: {len(packets)}")
        
        flows_by_attack = {attack: [] for attack in self.attack_mapping.keys()}
        
        flows = self.transformer.pcap2flow(str(pcap_file))
        
        for flow_key, flow_packets in tqdm(flows.items(), desc=f"Processing {pcap_file.name}"):
            if len(flow_packets) == 0:
                continue
            
            first_pkt = flow_packets[0]
            
            if not hasattr(first_pkt, 'IP'):
                continue
            
            src_ip = first_pkt['IP'].src
            dst_ip = first_pkt['IP'].dst
            
            src_port = 0
            dst_port = 0
            if hasattr(first_pkt, 'TCP'):
                src_port = first_pkt['TCP'].sport
                dst_port = first_pkt['TCP'].dport
            elif hasattr(first_pkt, 'UDP'):
                src_port = first_pkt['UDP'].sport
                dst_port = first_pkt['UDP'].dport
            
            matching_rows = flow_labels[
                (
                    ((flow_labels['Source IP'] == src_ip) & (flow_labels['Destination IP'] == dst_ip)) |
                    ((flow_labels['Source IP'] == dst_ip) & (flow_labels['Destination IP'] == src_ip))
                )
            ]
            
            if len(matching_rows) > 0:
                csv_label = matching_rows.iloc[0]['Label']
                attack_type = self.map_csv_label_to_attack_type(csv_label)
                
                if attack_type:
                    flows_by_attack[attack_type].append(flow_packets)
        
        for attack_type, flows_list in flows_by_attack.items():
            logger.info(f"  {attack_type}: {len(flows_list)} flows extracted")
        
        return flows_by_attack
    
    def process_csv_pcap_pair(self, csv_file: Path, pcap_file: Path, attack_counts: dict):
        """
        Process a CSV-PCAP pair and extract flows as images
        
        Args:
            csv_file: Path to CSV label file
            pcap_file: Path to PCAP file
            attack_counts: Dictionary tracking sample counts per attack type
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing pair: {csv_file.name} + {pcap_file.name}")
        logger.info(f"{'='*60}")
        
        flow_labels = self.load_csv_labels(csv_file)
        if flow_labels is None or len(flow_labels) == 0:
            logger.warning("No labeled flows found, skipping...")
            return
        
        flows_by_attack = self.extract_flows_from_pcap(pcap_file, flow_labels)
        
        for attack_type, flows_list in flows_by_attack.items():
            if len(flows_list) == 0:
                continue
            
            logger.info(f"\nConverting {attack_type} flows to images...")
            
            for flow_packets in tqdm(flows_list, desc=f"Converting {attack_type}"):
                try:
                    flow_image = self.transformer.flow2image(flow_packets)
                    
                    if np.random.random() < self.train_ratio:
                        save_dir = self.train_dir / attack_type
                    else:
                        save_dir = self.test_dir / attack_type
                    
                    sample_idx = attack_counts[attack_type]
                    filename = f"{attack_type}_{sample_idx:06d}.png"
                    filepath = save_dir / filename
                    
                    img = Image.fromarray(flow_image.astype(np.uint8))
                    img.save(filepath)
                    
                    attack_counts[attack_type] += 1
                    
                except Exception as e:
                    logger.warning(f"Error converting flow to image: {e}")
                    continue
    
    def process_all(self):
        """
        Process all CSV-PCAP pairs
        """
        logger.info("\n" + "="*60)
        logger.info("Starting CICIDS2017 Preprocessing (CSV-guided)")
        logger.info("="*60 + "\n")
        
        pairs = self.find_csv_pcap_pairs()
        
        if len(pairs) == 0:
            logger.error("No CSV-PCAP pairs found!")
            return
        
        logger.info(f"Found {len(pairs)} CSV-PCAP pairs\n")
        
        attack_counts = {attack: 0 for attack in self.attack_mapping.keys()}
        
        for csv_file, pcap_file in pairs:
            self.process_csv_pcap_pair(csv_file, pcap_file, attack_counts)
        
        logger.info("\n" + "="*60)
        logger.info("Preprocessing Complete!")
        logger.info("="*60)
        
        self.print_statistics()
    
    def print_statistics(self):
        """Print dataset statistics"""
        logger.info("\n" + "="*60)
        logger.info("Dataset Statistics")
        logger.info("="*60)
        
        for split in ['train', 'test']:
            split_dir = self.output_dir / split
            logger.info(f"\n{split.upper()} SET:")
            
            total = 0
            for attack_type in self.attack_mapping.keys():
                attack_dir = split_dir / attack_type
                if attack_dir.exists():
                    count = len(list(attack_dir.glob('*.png')))
                    logger.info(f"  {attack_type:20s}: {count:6d} samples")
                    total += count
            
            logger.info(f"  {'TOTAL':20s}: {total:6d} samples")
        
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CICIDS2017 dataset using CSV labels')
    parser.add_argument('--pcap-dir', type=str, required=True,
                        help='Directory containing PCAP files')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Directory containing CSV label files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed images')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    preprocessor = CICIDS2017Preprocessor(
        pcap_dir=args.pcap_dir,
        csv_dir=args.csv_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )
    
    preprocessor.process_all()


if __name__ == '__main__':
    main()
