#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser(
    description='Calculate gc percentage of 1 fasta file')
parser.add_argument('--fasta-file', "-ff", dest="fasta_file", type=str,
                    required=True, help='Path to fasta file')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()


def read_fasta(file_name):
    seqs = []
    seq = ''
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    seq = ''
            else:
                seq += line.strip()
        seqs.append(seq)

    return seqs


def get_gc_content(seq):
    seq = seq.upper()
    total_g = seq.count("G")
    total_c = seq.count("C")
    return ((total_g + total_c) / len(seq)) * 100


if __name__ == "__main__":
    seqs = read_fasta(args.fasta_file)
    gc_content = get_gc_content(seqs[0])
    print(f"The GC content is {gc_content}%")
