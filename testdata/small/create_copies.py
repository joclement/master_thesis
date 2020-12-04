from itertools import product
from pathlib import Path
from shutil import copyfile


for idx, csv_file in product(range(4), Path("./").glob("*.csv")):
    copyfile(csv_file, Path("./", f"{csv_file.stem}{idx}.csv"))
