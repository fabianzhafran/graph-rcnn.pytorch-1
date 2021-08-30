import json

if __name__ == "__main__":
    info = json.load(open("../datasets/vg_bm/VG-SGG-dicts.json", 'r'))
    print(info)