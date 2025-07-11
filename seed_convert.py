"""
    Expand the compressed IPv6 representation into 32 hexadecimal characters without colons.
"""

from IPy import IP
from tqdm import tqdm
import argparse

SEED_FILE = 'data/Seed_S1_10K.txt'


def normalizeIPv6(addrs):
    '''
        Expand the compressed IPv6 representation into 32 hexadecimal characters without colons.
    '''
    normal_addr = []
    for addr in tqdm(addrs):
        norm = IP(addr.strip()).strFullsize()
        addr_32hex = norm[:4]+norm[5:9]+norm[10:14]+norm[15:19]+norm[20:24]+norm[25:29]+norm[30:34]+norm[35:]
        normal_addr.append(addr_32hex)
    return normal_addr
# end_normalizeIPv6


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=SEED_FILE, type=str, required=False, help='IPv6 seed set file to be converted')
    args = parser.parse_args()
    
    # Read IPv6 addresses from a text file, each address on a separate line.
    with open(args.file, 'r') as f:
        all_addr_ = f.readlines()

    # Expand the compressed IPv6 representation into 32 hexadecimal characters without colons.
    addr_list = normalizeIPv6(all_addr_)
    addrs = '\n'.join(addr_list)           # Append a newline to each address.

    # Save the converted IPv6 addresses into a new file in the same directory, with '_32hex' added to the original file name. 
    f_addr_32hex = SEED_FILE.replace('.txt', '_32hex.txt')
    with open(f_addr_32hex, 'w') as f:
        f.writelines(addrs)
