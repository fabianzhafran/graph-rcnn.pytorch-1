from lib.data.refcoco_dataset import RefCOCO
from lib.data.flickr_dataset import Flickr30K

if __name__ == "__main__":
    flickr = Flickr30K()
    _, target, _, img_info = flickr[0]
    print("~~type(flickr[100])~~~")
    print(type(flickr[100]))
    print("~~img_info~~")
    print(img_info)
    print("~~target~~")
    print(target["extra_fields"])