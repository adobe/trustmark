# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import sys
import csv
import socket
import numpy as np
import json
import pickle  # python3.x
import time
from datetime import timedelta, datetime
from typing import Any, List, Tuple, Union
import subprocess
import struct
import errno
from pprint import pprint
import glob
from threading import Thread


def welcome_message():
    """
    get welcome message including hostname and command line arguments
    """
    hostname = socket.gethostname()
    all_args = ' '.join(sys.argv)
    out_text = 'On server {}: {}\n'.format(hostname, all_args)
    return out_text


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __init__(self, dict_to_convert=None):
        if dict_to_convert is not None:
            for key, val in dict_to_convert.items():
                self[key] = val
                
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def get_time_id_str():
    """
    returns a string with DDHHM format, where M is the minutes cut to the tenths
    """
    now = datetime.now()
    time_str = "{:02d}{:02d}{:02d}".format(now.day, now.hour, now.minute)
    time_str = time_str[:-1]
    return time_str


def time_format(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    m, h, s = int(m), int(h), int(s)

    if m == 0 and h == 0:
        return "{}s".format(s)
    elif h == 0:
        return "{}m{}s".format(m, s)
    else:
        return "{}h{}m{}s".format(h, m, s)


def get_all_files(dir_path, trim=0, extension=''):
    """
    Recursively get list of all files in the given directory
    trim = 1 : trim the dir_path from results, 0 otherwise
    extension: get files with specific format
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    if trim == 1:  # trim dir_path from results
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        trim_len = len(dir_path)
        file_paths = [x[trim_len:] for x in file_paths]

    if extension:  # select only file with specific extension
        extension = extension.lower()
        tlen = len(extension)
        file_paths = [x for x in file_paths if x[-tlen:] == extension]

    return file_paths  # Self-explanatory.


def get_all_dirs(dir_path, trim=0):
    """
    Recursively get list of all directories in the given directory
    excluding the '.' and '..' directories
    trim = 1 : trim the dir_path from results, 0 otherwise
    """
    out = []
    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for dirname in directories:
            # Join the two strings in order to form the full filepath.
            dir_full = os.path.join(root, dirname)
            out.append(dir_full)  # Add it to the list.

    if trim == 1:  # trim dir_path from results
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        trim_len = len(dir_path)
        out = [x[trim_len:] for x in out]

    return out


def read_list(file_path, delimeter=' ', keep_original=True):
    """
    read list column wise
    deprecated, should use pandas instead
    """
    out = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimeter)
        for row in reader:
            out.append(row)
    out = zip(*out)

    if not keep_original:
        for col in range(len(out)):
            if out[col][0].isdigit():  # attempt to convert to numerical array
                out[col] = np.array(out[col]).astype(np.int64)

    return out


def save_pickle2(file_path, **kwargs):
    """
    save variables to file (using pickle)
    """
    # check if any variable is a dict
    var_count = 0
    for key in kwargs:
        var_count += 1
        if isinstance(kwargs[key], dict):
            sys.stderr.write('Opps! Cannot write a dictionary into pickle')
            sys.exit(1)
    with open(file_path, 'wb') as f:
        pickler = pickle.Pickler(f, -1)
        pickler.dump(var_count)
        for key in kwargs:
            pickler.dump(key)
            pickler.dump(kwargs[key])


def load_pickle2(file_path, varnum=0):
    """
    load variables that previously saved using self.save()
    varnum : number of variables u want to load (0 mean it will load all)
    Note: if you are loading class instance(s), you must have it defined in advance
    """
    with open(file_path, 'rb') as f:
        pickler = pickle.Unpickler(f)
        var_count = pickler.load()
        if varnum:
            var_count = min([var_count, varnum])
        out = {}
        for i in range(var_count):
            key = pickler.load()
            out[key] = pickler.load()

    return out


def save_pickle(path, obj):
    """
    simple method to save a picklable object
    :param path: path to save
    :param obj: a picklable object
    :return: None
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    """
    load a pickled object
    :param path: .pkl path
    :return: the pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_new_dir(dir_path, remove_existing=False, mode=511):
    """note: default mode in ubuntu is 511"""
    if not os.path.exists(dir_path):
        try:
            if mode == 777:
                oldmask = os.umask(000)
                os.makedirs(dir_path, 0o777)
                os.umask(oldmask)
            else:
                os.makedirs(dir_path, mode)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
                pass
            else:
                raise
    if remove_existing:
        for file_obj in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_obj)
            if os.path.isfile(file_path):
                os.unlink(file_path)


def get_latest_file(root, pattern):
    """
    get the latest file in a directory that match the provided pattern
    useful for getting the last checkpoint
    :param root: search directory
    :param pattern: search pattern containing 1 wild card representing a number e.g. 'ckpt_*.tar'
    :return: full path of the file with largest number in wild card, None if not found
    """
    out = None
    parts = pattern.split('*')
    max_id = - np.inf
    for path in glob.glob(os.path.join(root, pattern)):
        id_ = os.path.basename(path)
        for part in parts:
            id_ = id_.replace(part, '')
        try:
            id_ = int(id_)
            if id_ > max_id:
                max_id = id_
                out = path
        except:
            continue
    return out


class Locker(object):
    """place a lock file in specified location
    useful for distributed computing"""

    def __init__(self, name='lock.txt', mode=511):
        """INPUT: name default file name to be created as a lock
                  mode if a directory has to be created, set its permission to mode"""
        self.name = name
        self.mode = mode

    def lock(self, path):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write('progress')

    def finish(self, path):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write('finish')

    def customise(self, path, text):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write(text)

    def is_locked(self, path):
        out = False
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            text = open(check_path, 'r').readline().strip()
            out = True if text == 'progress' else False
        return out

    def is_finished(self, path):
        out = False
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            text = open(check_path, 'r').readline().strip()
            out = True if text == 'finish' else False
        return out
        
    def is_locked_or_finished(self, path):
        return self.is_locked(path) | self.is_finished(path)

    def clean(self, path):
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            try:
                os.remove(check_path)
            except Exception as e:
                print('Unable to remove %s: %s.' % (check_path, e))


class ProgressBar(object):
    """show progress"""

    def __init__(self, total, increment=5):
        self.total = total
        self.point = self.total / 100.0
        self.increment = increment
        self.interval = int(self.total * self.increment / 100)
        self.milestones = list(range(0, total, self.interval)) + [self.total, ]
        self.id = 0

    def show_progress(self, i):
        if i >= self.milestones[self.id]:
            while i >= self.milestones[self.id]:
                self.id += 1
            sys.stdout.write("\r[" + "=" * int(i / self.interval) +
                             " " * int((self.total - i) / self.interval) + "]" + str(int((i + 1) / self.point)) + "%")
            sys.stdout.flush()


class Timer(object):

    def __init__(self):
        self.start_t = time.time()
        self.last_t = self.start_t

    def time(self, lap=False):
        end_t = time.time()
        if lap:
            out = timedelta(seconds=int(end_t - self.last_t))  # count from last stop point
        else:
            out = timedelta(seconds=int(end_t - self.start_t))  # count from beginning
        self.last_t = end_t
        return out


class ExThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(ExThread, self).join()
        if self.exc:
            raise RuntimeError('Exception in thread.') from self.exc
        return self.ret


def get_gpu_free_mem():
    """return a list of free GPU memory"""
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8") .split('\n')

    out = []
    for i in range(len(out_list)):
        item = out_list[i]
        if item.strip() == 'FB Memory Usage':
            free_mem = int(out_list[i + 3].split(':')[1].strip().split(' ')[0])
            out.append(free_mem)
    return out


def float2hex(x):
    """
    x: a vector
    return: x in hex
    """
    f = np.float32(x)
    out = ''
    if f.size == 1:  # just a single number
        f = [f, ]
    for e in f:
        h = hex(struct.unpack('<I', struct.pack('<f', e))[0])
        out += h[2:].zfill(8)
    return out


def hex2float(x):
    """
    x: a string with len divided by 8
    return x as array of float32
    """
    assert len(x) % 8 == 0, 'Error! string len = {} not divided by 8'.format(len(x))
    l = len(x) / 8
    out = np.empty(l, dtype=np.float32)
    x = [x[i:i + 8] for i in range(0, len(x), 8)]
    for i, e in enumerate(x):
        out[i] = struct.unpack('!f', e.decode('hex'))[0]
    return out


def nice_print(inputs, stream=sys.stdout):
    """print a list of string to file stream"""
    if type(inputs) is not list:
        tstrings = inputs.split('\n')
        pprint(tstrings, stream=stream)
    else:
        for string in inputs:
            nice_print(string, stream=stream)
    stream.flush()
