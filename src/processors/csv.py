import csv
header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
def save_rows(rows: list, out_file: str):
    with open(out_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row in rows:
            writer.writerow(row)