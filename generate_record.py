import argparse
from src import util, record_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Process available datasets',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-c', '--csv')
    parser.add_argument('-r', '--record')
    parser.add_argument('-i', '--img_dir')
    parser.add_argument('-l', '--labels')
    
    args = parser.parse_args()
    
    record_csv.create_record_csv(
        args.csv, 
        args.img_dir, 
        args.record, 
        args.labels
    )