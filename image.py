from PIL import Image
import os

def crop_annotations(source_dir, target_dir, ann, fi):
    [name, ext] = ann["file_name"].split(".") 
    im = Image.open(f"{source_dir}\{ann['file_name']}")
    
    i = 0
    for a in ann["annotations"]:
        box = a["bbox"]
        part = im.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        print(name)
        for c in a["classes"]:
            if os.path.isdir(f"{target_dir}/{c}") == False:
                os.mkdir(f"{target_dir}/{c}")

            part.save(f"{target_dir}/{c}/{fi}_{i}.{ext}")
            
        i += 1