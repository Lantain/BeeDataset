import csv

header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']

def annotation_to_rows(ann):
    rows = list()
    height = ann["height"]
    width = ann["width"]
    filename = ann['file_name']

    for a in ann["annotations"]:
        xmin = a['bbox']['xmin']
        xmax = a['bbox']['xmax']
        ymin = a['bbox']['ymin']
        ymax = a['bbox']['ymax']

        for class_name in a['classes']:
            row = [filename, class_name, width, height, xmin, ymin, xmax, ymax]
            rows.append(row)
    
    return rows

def generate_csv_from_annotation(ann, out_dir):
    filename = ann['file_name']
    [name, ext] = filename.split(".") 

    with open(f"{out_dir}/{name}.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rows = annotation_to_rows(ann)
        for row in rows:
            writer.writerow(row)

def generate_csv_from_annotation_set(anns, out_file):
    with open(out_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ann in anns:
            rows = annotation_to_rows(ann)
            for row in rows:
                writer.writerow(row)

