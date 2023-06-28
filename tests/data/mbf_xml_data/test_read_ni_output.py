import os
from dotenv import load_dotenv
import pprint


import sys
sys.path.append(os.getcwd())

# print(type(sys.path))
pprint.pprint(sys.path, width=1)
print(os.getcwd())

from src.core.utils import fs


# Load environment variables from .env file
load_dotenv()
sys.path.append(os.getcwd())


def main():
    if load_dotenv() is False:
        print("Failed to load .env file")
        return
    
    example_case_file_dir = os.getenv("CASE_FILE_DATA_DIR")
    if example_case_file_dir is None:
        print("Failed to load CASE_FILE_DATA_DIR from .env file")
        return
    
    case_files = fs.find_in_dir_with_ext(example_case_file_dir, ".xml")
    if len(case_files) == 0:
        print("No case files found in directory")
        return
    
    case_file_path = case_files[0]
    print(f"Reading case file: {os.path.basename(case_file_path)}")
    

if __name__ == "__main__":
    main()