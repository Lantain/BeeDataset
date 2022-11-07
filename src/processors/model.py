import requests
import tarfile

def download_model(model_name):
    model_url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"
    r = requests.get(model_url, allow_redirects=True)
    open(f"out/{model_name}.tar.gz", 'wb').write(r.content)

def decompress_model(model_name):
    file = tarfile.open(f'out/{model_name}.tar.gz')
    file.extractall(f'out/')
    file.close()