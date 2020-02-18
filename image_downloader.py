from PIL import Image
import urllib.request

"""
Fill this array with URLs pointing to images.
"""
url_list = [
]

for (i, url) in enumerate(url_list):
    urllib.request.urlretrieve(url, f"./train_data/gan2/real/{i}.jpg")

    image = Image.open(f"./train_data/gan2/real/{i}.jpg")
    newimg = image.resize((400, 400))
    newimg.save(f"./train_data/gan2/real/{i}.jpg")
