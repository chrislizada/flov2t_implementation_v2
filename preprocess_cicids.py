import argparse
import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from data.packet2patch import Packet2PatchTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CICIDS2017Preprocessor:
    """
    Preprocess CICIDS2017 PCAP files into images using Packet2Patch transformation
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 attack_types: list = None,
                 train_ratio: float = 0.8):
        """
        Args:
            input_dir: Directory containing raw PCAP files
            output_dir: Directory to save processed images
            attack_types: List of attack types to process
            train_ratio: Ratio of training vs test split
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        if attack_types is None:
            self.attack_types = [
                'Botnet',
                'DoS-Slowloris',
                'DoS-Goldeneye',
                'DoS-Hulk',
                'SSH-BruteForce',
                'Web-SQL',
                'Web-XSS',
                'Web-Bruteforce'
            ]
        else:
            self.attack_types = attack_types
        
        self.transformer = Packet2PatchTransformer(
            max_packets=196,
            patch_size=16,
            net_header_bytes=20,
            trans_header_bytes=20,
            payload_bytes=216
        )
        
        self.train_dir = self.output_dir / 'train'
        self.test_dir = self.output_dir / 'test'
        
        for attack_type in self.attack_types:
            (self.train_dir / attack_type).mkdir(parents=True, exist_ok=True)
            (self.test_dir / attack_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessor initialized")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Attack types: {self.attack_types}")
    
    def find_pcap_files(self, attack_type: str) -> list:
        """
        Find PCAP files for specific attack type
        
        Args:
            attack_type: Attack category name
        
        Returns:
            List of PCAP file paths
        """
        patterns = [
            f"*{attack_type}*.pcap",
            f"*{attack_type.lower()}*.pcap",
            f"*{attack_type.upper()}*.pcap"
        ]
        
        pcap_files = []
        for pattern in patterns:
            pcap_files.extend(glob.glob(str(self.input_dir / pattern)))
        
        return list(set(pcap_files))
    
    def process_attack_type(self, attack_type: str, max_samples_per_type: int = None):
        """
        Process all PCAP files for a specific attack type
        
        Args:
            attack_type: Attack category name
            max_samples_per_type: Maximum samples to process (for testing)
        """
        logger.info(f"Processing attack type: {attack_type}")
        
        pcap_files = self.find_pcap_files(attack_type)
        
        if not pcap_files:
            logger.warning(f"No PCAP files found for {attack_type}")
            return
        
        logger.info(f"Found {len(pcap_files)} PCAP file(s) for {attack_type}")
        
        total_flows = 0
        train_count = 0
        test_count = 0
        
        for pcap_file in pcap_files:
            logger.info(f"Processing: {pcap_file}")
            
            try:
                flows = self.transformer.pcap2flow(pcap_file)
                
                for flow_idx, (flow_key, packets) in enumerate(tqdm(flows.items(), desc=f"Converting {attack_type}")):
                    if max_samples_per_type and total_flows >= max_samples_per_type:
                        break
                    
                    try:
                        flow_image = self.transformer.flow2image(packets)
                        
                        if np.random.random() < self.train_ratio:
                            save_dir = self.train_dir / attack_type
                            train_count += 1
                        else:
                            save_dir = self.test_dir / attack_type
                            test_count += 1
                        
                        filename = f"{attack_type}_{total_flows:06d}.png"
                        filepath = save_dir / filename
                        
                        img = Image.fromarray(flow_image.astype(np.uint8))
                        img.save(filepath)
                        
                        total_flows += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing flow {flow_idx}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing PCAP file {pcap_file}: {e}")
                continue
        
        logger.info(f"Completed {attack_type}: {total_flows} total ({train_count} train, {test_count} test)")
    
    def process_all(self, max_samples_per_type: int = None):
        """
        Process all attack types
        
        Args:
            max_samples_per_type: Maximum samples per attack type (for testing)
        """
        logger.info("Starting preprocessing...")
        
        for attack_type in self.attack_types:
            self.process_attack_type(attack_type, max_samples_per_type)
        
        logger.info("Preprocessing complete!")
        
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
            for attack_type in self.attack_types:
                attack_dir = split_dir / attack_type
                if attack_dir.exists():
                    count = len(list(attack_dir.glob('*.png')))
                    logger.info(f"  {attack_type:20s}: {count:6d} samples")
                    total += count
            
            logger.info(f"  {'TOTAL':20s}: {total:6d} samples")
        
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CICIDS2017 dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing PCAP files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed images')
    parser.add_argument('--attack-types', nargs='+', default=None,
                        help='Attack types to process (default: all 8 types)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per attack type (for testing)')
    
    args = parser.parse_args()
    
    preprocessor = CICIDS2017Preprocessor(
        input_dir=args.input,
        output_dir=args.output,
        attack_types=args.attack_types,
        train_ratio=args.train_ratio
    )
    
    preprocessor.process_all(max_samples_per_type=args.max_samples)


if __name__ == '__main__':
    main()
