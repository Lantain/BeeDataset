import argparse
from src.datasets import remo
from src.datasets import record

SPLIT_RATIO = .8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Process available datasets',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-t', '--type')
    parser.add_argument('-f', '--file')
    parser.add_argument('-s', '--src_dir')
    parser.add_argument('-o', '--out_dir')
    
    args = parser.parse_args()

    if args.type == 'remo':
        remo.generate_dataset(f'{args.dir}/remo.json', f'{args.src_dir}', args.out_dir)
    elif args.type == 'record':
        record.generate_dataset(args.file, args.out_dir)