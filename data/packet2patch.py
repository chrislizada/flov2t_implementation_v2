import numpy as np
from typing import List, Dict, Tuple
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Packet2PatchTransformer:
    """
    Implements the Packet2Patch transformation from the FLoV2T paper.
    
    Transforms raw network traffic into 224x224 images:
    - Each packet becomes a 16x16 patch
    - Structure: [20B net_header | 20B trans_header | 216B payload]
    - Up to 196 packets per flow (14x14 grid)
    """
    
    def __init__(self, 
                 max_packets: int = 196,
                 patch_size: int = 16,
                 net_header_bytes: int = 20,
                 trans_header_bytes: int = 20,
                 payload_bytes: int = 216):
        """
        Args:
            max_packets: Maximum packets per flow (default: 196 for 14x14 grid)
            patch_size: Size of each patch (default: 16x16)
            net_header_bytes: Bytes for network layer header
            trans_header_bytes: Bytes for transport layer header
            payload_bytes: Bytes for payload/extensions
        """
        self.max_packets = max_packets
        self.patch_size = patch_size
        self.net_header_bytes = net_header_bytes
        self.trans_header_bytes = trans_header_bytes
        self.payload_bytes = payload_bytes
        self.bytes_per_patch = patch_size * patch_size  # 256 bytes
        
        assert net_header_bytes + trans_header_bytes + payload_bytes == self.bytes_per_patch, \
            f"Header + payload must equal {self.bytes_per_patch} bytes"
        
    def pcap2flow(self, pcap_file: str) -> Dict[Tuple, List]:
        """
        Split PCAP file into flows based on 5-tuple.
        
        Args:
            pcap_file: Path to PCAP file
            
        Returns:
            Dictionary mapping 5-tuple to list of packets
        """
        flows = defaultdict(list)
        
        try:
            packets = rdpcap(pcap_file)
            logger.info(f"Read {len(packets)} packets from {pcap_file}")
            
            for pkt in packets:
                if IP in pkt:
                    # Extract 5-tuple
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                    
                    src_port = 0
                    dst_port = 0
                    if TCP in pkt:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    elif UDP in pkt:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                    
                    # Create bidirectional flow key (sorted to group both directions)
                    flow_key = tuple(sorted([
                        (src_ip, src_port),
                        (dst_ip, dst_port)
                    ]) + [proto])
                    
                    flows[flow_key].append(pkt)
            
            logger.info(f"Created {len(flows)} flows")
            return flows
            
        except Exception as e:
            logger.error(f"Error reading PCAP file: {e}")
            return {}
    
    def packet2patch(self, packet) -> np.ndarray:
        """
        Convert single packet to 16x16 patch.
        
        Structure (256 bytes total):
        - Bytes 0-19: Network layer header (IP)
        - Bytes 20-39: Transport layer header (TCP/UDP)
        - Bytes 40-255: Payload + extensions
        
        Args:
            packet: Scapy packet object
            
        Returns:
            16x16 numpy array
        """
        patch = np.zeros(self.bytes_per_patch, dtype=np.uint8)
        
        try:
            # Extract raw bytes
            raw_bytes = bytes(packet)
            
            # Network layer header (first 20 bytes of IP header)
            if IP in packet:
                ip_bytes = bytes(packet[IP])[:self.net_header_bytes]
                patch[:len(ip_bytes)] = list(ip_bytes)
            
            # Transport layer header (20 bytes)
            offset = self.net_header_bytes
            if TCP in packet:
                tcp_bytes = bytes(packet[TCP])[:self.trans_header_bytes]
                patch[offset:offset+len(tcp_bytes)] = list(tcp_bytes)
            elif UDP in packet:
                udp_bytes = bytes(packet[UDP])[:self.trans_header_bytes]
                patch[offset:offset+len(udp_bytes)] = list(udp_bytes)
            
            # Payload (remaining bytes)
            offset = self.net_header_bytes + self.trans_header_bytes
            if packet.haslayer('Raw'):
                payload = bytes(packet['Raw'])[:self.payload_bytes]
                patch[offset:offset+len(payload)] = list(payload)
            
        except Exception as e:
            logger.warning(f"Error converting packet to patch: {e}")
        
        # Reshape to 16x16
        return patch.reshape(self.patch_size, self.patch_size)
    
    def flow2image(self, flow_packets: List) -> np.ndarray:
        """
        Concatenate patches into 224x224 RGB image.
        
        Args:
            flow_packets: List of packets in the flow
            
        Returns:
            224x224x3 numpy array (RGB image)
        """
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Take first max_packets
        packets = flow_packets[:self.max_packets]
        
        # Pad if needed (repeat handshake/first packet)
        if len(packets) < self.max_packets and len(packets) > 0:
            padding_packet = packets[0]
            packets = packets + [padding_packet] * (self.max_packets - len(packets))
        elif len(packets) == 0:
            # Empty flow - return zero image
            return image
        
        # Arrange patches in 14x14 grid
        patches_per_row = 14
        for idx, packet in enumerate(packets):
            patch = self.packet2patch(packet)
            
            # Calculate position in grid
            row_idx = idx // patches_per_row
            col_idx = idx % patches_per_row
            
            # Place patch in image (replicate to RGB)
            row_start = row_idx * self.patch_size
            col_start = col_idx * self.patch_size
            
            for c in range(3):  # RGB channels
                image[row_start:row_start+self.patch_size,
                      col_start:col_start+self.patch_size, c] = patch
        
        return image
    
    def transform_pcap(self, pcap_file: str, label: int = None) -> List[Tuple[np.ndarray, int]]:
        """
        End-to-end transformation: PCAP -> Images
        
        Args:
            pcap_file: Path to PCAP file
            label: Optional label for all flows in this PCAP
            
        Returns:
            List of (image, label) tuples
        """
        flows = self.pcap2flow(pcap_file)
        
        images_labels = []
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) > 0:
                image = self.flow2image(flow_packets)
                images_labels.append((image, label))
        
        logger.info(f"Transformed {len(images_labels)} flows to images")
        return images_labels
    
    def transform_flow_dict(self, flows: Dict, labels: Dict = None) -> List[Tuple[np.ndarray, int]]:
        """
        Transform pre-extracted flows to images.
        
        Args:
            flows: Dictionary of flow_key -> packets
            labels: Dictionary of flow_key -> label
            
        Returns:
            List of (image, label) tuples
        """
        images_labels = []
        
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) > 0:
                image = self.flow2image(flow_packets)
                label = labels.get(flow_key, -1) if labels else -1
                images_labels.append((image, label))
        
        return images_labels


if __name__ == "__main__":
    # Test the transformer
    logging.basicConfig(level=logging.INFO)
    
    transformer = Packet2PatchTransformer()
    print(f"Packet2Patch Transformer initialized")
    print(f"Max packets: {transformer.max_packets}")
    print(f"Patch size: {transformer.patch_size}x{transformer.patch_size}")
    print(f"Bytes per patch: {transformer.bytes_per_patch}")
