# legacy_fix/aifc.py
class Error(Exception):
    pass

def open(f, mode=None):
    raise Error("aifc is not supported in Python 3.13+, but we are bypassing the import error.")