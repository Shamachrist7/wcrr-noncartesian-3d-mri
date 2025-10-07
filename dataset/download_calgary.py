import gdown
from tqdm import tqdm
import zipfile
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default=None)  # Your root directory
args = parser.parse_args()
root = args.root

train_path = os.path.join(root, "Train")
val_path = os.path.join(root, "Val")
test_path_12coil = os.path.join(root, "Test/12coil")
test_path_32coil = os.path.join(root, "Test/32coil")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path_12coil, exist_ok=True)
os.makedirs(test_path_32coil, exist_ok=True)

url1 = "https://drive.google.com/uc?id=1IJN0O_Y4DXtaKzvkPfPp_8CO6P0wPUGz"
out_path1 = os.path.join(root, "Train_12coil_part1.zip")
print("Downloading the 12-coil Training set, Part 1 ...")
gdown.download(url1, out_path1, quiet=False)
with zipfile.ZipFile(out_path1, "r") as zip_ref:
    zip_ref.extractall(train_path)
print("Training set part 1 downloaded!")

url2 = "https://drive.google.com/uc?id=1IJN0O_Y4DXtaKzvkPfPp_8CO6P0wPUGz"
out_path2 = os.path.join(root, "Train_12coil_part2.zip")
print("Downloading the 12-coil Training set, Part 2 ...")
gdown.download(url2, out_path2, quiet=False)
with zipfile.ZipFile(out_path2, "r") as zip_ref:
    zip_ref.extractall(train_path)
print("Training set part 2 downloaded!")

url3 = "https://drive.google.com/uc?id=1IJN0O_Y4DXtaKzvkPfPp_8CO6P0wPUGz"
out_path3 = os.path.join(root, "Val_12coil.zip")
print("Downloading the 12-coil Validation set ...")
gdown.download(url3, out_path3, quiet=False)
with zipfile.ZipFile(out_path3, "r") as zip_ref:
    zip_ref.extractall(val_path)
print("Validation set downloaded!")

url4 = "https://drive.google.com/uc?id=1gJaz9fI9pmcFXEm7ntUMc6QTElmHta4H"
out_path4 = os.path.join(root, "Test_12and32coil.zip")
print("Downloading the 12-coil and 32-coil Testing set ...")
gdown.download(url4, out_path4, quiet=False)
with zipfile.ZipFile(out_path4, "r") as zip_ref:
    zip_ref.extractall(os.path.join(root, "Test"))
print("Testing set downloaded!")
