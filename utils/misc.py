import inspect
from os.path import basename
from shutil import copyfile


def copy_object_sourcefile(source_obj, dest_dir):
    x = source_obj.__class__
    while x != object.__class__ and 'torch' not in str(x):
        path = inspect.getsourcefile(x)
        copyfile(path, dest_dir + '/' + basename(path))
        x = x.__base__
