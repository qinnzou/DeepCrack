import os
from os import path
from shutil import copyfile
import errno

__all__ = ['split', 'make', 'copy_to_dir', 'process', 'write_file']


def split(directory):
    """Splits a full filename path into its directory path, name and extension

    Args:
        directory (str): Directory to split.

    Returns:
        tuple: (Directory name, filename, extension)
    """
    directory = process(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


def make(directory):
    """Make a new directory

    Args:
        directory (str): Directory to make.
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def copy_to_dir(file, directory):
    """Copies a file to a directory

    Args:
        file (str): File to copy.
        directory (str): Directory to copy file to.
    """
    file_path = path.join(directory, path.basename(file))
    copyfile(file, file_path)


def process(directory, create=False):
    """Expands home path, finds absolute path and creates directory (if create is True).

    Args:
        directory (str): Directory to process.
        create (bool, optional): If True, it creates the directory.

    Returns:
        str: The processed directory.
    """
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    if create:
        make(directory)
    return directory


def write_file(contents, filename, directory=".", append=False):
    """Writes contents to file.

    Args:
        contents (str): Contents to write to file.
        filename (str): File to write contents to.
        directory (str, optional): Directory to put file in.
        append (bool, optional): If True and file exists, it appends contents.

    Returns:
        str: Full path to file.
    """
    full_name = path.join(process(directory), filename)
    mode = "a" if append else "w"
    with open(full_name, mode) as file_handle:
        file_handle.write(contents)
    return full_name
