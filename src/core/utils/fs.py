import glob
import os
from pathlib import Path
from typing import Union, List


def prepare_directories(directory_path: str):
    run_id = 0
    directory_name = "run_{:03}".format(run_id)
    if os.path.exists(os.path.join(directory_path, directory_name)):
        # Directory already exists, increment the id
        run_id += 1
        while os.path.exists(os.path.join(directory_path, "run_{:03}".format(run_id))):
            run_id += 1
        directory_name = "run_{:03}".format(run_id)
    directory_path = os.path.join(directory_path, directory_name)
    os.makedirs(directory_path, exist_ok=True)
    os.makedirs(os.path.join(directory_path, "checkpoints"), exist_ok=True)
    return directory_path


def has_ext(file_name: str, extensions: Union[str, List[str]]) -> bool:
    if not isinstance(extensions, (str, list)):
        raise ValueError("Argument extensions must be either string or list of strings")
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = set(map(str.lower, extensions))

    name, ext = os.path.splitext(file_name)
    return ext.lower() in extensions


def find_in_dir(dir_path: str) -> List[str]:
    return [os.path.join(dir_path, file_name) for file_name in sorted(os.listdir(dir_path))]


def id_from_file_path(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def find_subdirectories_in_dir(dir_path: str) -> List[str]:
    """
    Retrieve list of subdirectories (non-recursive) in the given directory.
    Args:
        dir_path: Target directory name

    Returns:
        Sorted list of absolute paths to directories
    """
    all_entries = find_in_dir(dir_path)
    return [entry for entry in all_entries if os.path.isdir(entry)]


def find_in_dir_with_ext(dir_path: str, extensions: Union[str, List[str]]) -> List[str]:
    return [os.path.join(dir_path, file_name) for file_name in sorted(os.listdir(dir_path)) if has_ext(file_name, extensions)]


def change_extension(file_name: Union[str, Path], new_ext: str) -> Union[str, Path]:
    if isinstance(file_name, str):
        return os.path.splitext(file_name)[0] + new_ext
    elif isinstance(file_name, Path):
        if new_ext[0] != ".":
            new_ext = "." + new_ext
        return file_name.with_suffix(new_ext)
    else:
        raise RuntimeError(
            f"Received input argument `file_name` for unsupported type {type(file_name)}. Argument must be string or Path."
        )


def is_file(file_path: Union[str, Path]) -> bool:
    if isinstance(file_path, str):
        return os.path.isfile(file_path)
    elif isinstance(file_path, Path):
        return file_path.is_file()
    else:
        raise RuntimeError(
            f"Received input argument `file_path` for unsupported type {type(file_path)}. Argument must be string or Path."
        )


def auto_file(filename: str, where: str = ".") -> str:
    """Get a full path to file using its name.
    This function recursively search for matching filename in @where and returns single match.
    :param where:
    :param filename:
    :return:
    """
    if os.path.isabs(filename):
        return filename

    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return prob

    files = list(glob.iglob(os.path.join(where, "**", filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError("Given file could not be found with recursive search:" + filename)

    if len(files) > 1:
        raise FileNotFoundError(
            "More than one file matches given filename. Please specify it explicitly:\n" + "\n".join(files)
        )

    return files[0]
