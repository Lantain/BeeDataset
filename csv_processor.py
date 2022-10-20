import csv

header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']

def generate_csv_from_annotation(ann, out_dir):
    filename = ann['file_name']
    [name, ext] = filename.split(".") 
    height = ann["height"]
    width = ann["width"]

    with open(f"{out_dir}/{name}.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for a in ann["annotations"]:
            xmin = a['bbox']['xmin']
            xmax = a['bbox']['xmax']
            ymin = a['bbox']['ymin']
            ymax = a['bbox']['ymax']

            for c in a['classes']:
                row = [filename, c, width, height, xmin, ymin, xmax, ymax]
                writer.writerow(row)

def generate_csv_from_annotation_set(anns, out_file):
    with open(out_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ann in anns:
            filename = ann['file_name']
            height = ann["height"]
            width = ann["width"]

            for a in ann["annotations"]:
                xmin = a['bbox']['xmin']
                xmax = a['bbox']['xmax']
                ymin = a['bbox']['ymin']
                ymax = a['bbox']['ymax']

                for c in a['classes']:
                    row = [filename, c, width, height, xmin, ymin, xmax, ymax]
                    writer.writerow(row)

