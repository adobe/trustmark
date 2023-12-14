# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from typing import Any, Optional, Union
from pathlib import Path
import os
import io
import lmdb
import pickle
import gzip
import bz2
import lzma
import shutil
from tqdm import tqdm
import pandas as pd 
import numpy as np
from numpy import ndarray
import time
import torch
from torch import Tensor
from distutils.dir_util import copy_tree
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _default_encode(data: Any, protocol: int) -> bytes:
    return pickle.dumps(data, protocol=protocol)


def _ascii_encode(data: str) -> bytes:
    return data.encode("ascii")


def _default_decode(data: bytes) -> Any:
    return pickle.loads(data)


def _default_decompress(data: bytes) -> bytes:
    return data


def _decompress(compression: Optional[str]):
    if compression is None:
        _decompress = _default_decompress
    elif compression == "gzip":
        _decompress = gzip.decompress
    elif compression == "bz2":
        _decompress = bz2.decompress
    elif compression == "lzma":
        _decompress = lzma.decompress
    else:
        raise ValueError(f"Unknown compression algorithm: {compression}")

    return _decompress


class BaseLMDB(object):
    _database = None
    _protocol = None
    _length = None

    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
        pre_open: bool = False,
        compression: Optional[str] = None
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism.
        :param pre_open: If set to True, the first iterations will be faster, but it will raise error when doing multi-gpu training. If set to False, the database will open when you will retrieve the first item.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead
        self.pre_open = pre_open
        self._decompress = _decompress(compression)
        self._has_fetched_an_item = False

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The set of available keys.
        """
        if self._protocol is None:
            self._protocol = self._get(
                item="protocol",
                encode_key=_ascii_encode,
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._protocol

    @property
    def keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            item="keys",
            encode_key=lambda key: _default_encode(key, protocol=protocol),
            decompress_value=_default_decompress,
            decode_value=_default_decode,
        )
        return keys

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """
        Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched_an_item = True
        if not isinstance(item, list):
            item = self._get(
                item=item,
                encode_key=self._encode_key,
                decompress_value=self._decompress_value,
                decode_value=self._decode_value,
            )
        else:
            item = self._gets(
                items=item,
                encode_keys=self._encode_keys,
                decompress_values=self._decompress_values,
                decode_values=self._decode_values,
            )
        return item

    def _get(self, item, encode_key, decompress_value, decode_value):
        """
        Instantiates a transaction and its associated cursor to fetch an item.

        :param item: A key.
        :param encode_key:
        :param decode_value:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                item = self._fetch(
                    cursor=cursor,
                    key=item,
                    encode_key=encode_key,
                    decompress_value=decompress_value,
                    decode_value=decode_value,
                )
        self._keep_database()
        return item

    def _gets(self, items, encode_keys, decompress_values, decode_values):
        """
        Instantiates a transaction and its associated cursor to fetch a list of items.

        :param items: A list of keys.
        :param encode_keys:
        :param decode_values:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                items = self._fetchs(
                    cursor=cursor,
                    keys=items,
                    encode_keys=encode_keys,
                    decompress_values=decompress_values,
                    decode_values=decode_values,
                )
        self._keep_database()
        return items

    def _fetch(self, cursor, key, encode_key, decompress_value, decode_value):
        """
        Retrieve a value given a key.

        :param cursor:
        :param key: A key.
        :param encode_key:
        :param decode_value:
        :return: A value.
        """
        key = encode_key(key)
        value = cursor.get(key)
        value = decompress_value(value)
        value = decode_value(value)
        return value

    def _fetchs(self, cursor, keys, encode_keys, decompress_values, decode_values):
        """
        Retrieve a list of values given a list of keys.

        :param cursor:
        :param keys: A list of keys.
        :param encode_keys:
        :param decode_values:
        :return: A list of values.
        """
        keys = encode_keys(keys)
        _, values = list(zip(*cursor.getmulti(keys)))
        values = decompress_values(values)
        values = decode_values(values)
        return values

    def _encode_key(self, key: Any) -> bytes:
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return pickle.dumps(key, protocol=self.protocol)

    def _encode_keys(self, keys: list) -> list:
        """
        Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._encode_key(key=key) for key in keys]

    def _decompress_value(self, value: bytes) -> bytes:
        return self._decompress(value)

    def _decompress_values(self, values: list) -> list:
        return [self._decompress_value(value=value) for value in values]

    def _decode_value(self, value: bytes) -> Any:
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return pickle.loads(value)

    def _decode_values(self, values: list) -> list:
        """
        Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._decode_value(value=value) for value in values]

    def _keep_database(self):
        """
        Checks if the database must be deleted.

        :return:
        """
        if not self.pre_open and not self._has_fetched_an_item:
            del self.database

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        del self.database

    @staticmethod
    def write(data_lst, indir, outdir):
        raise NotImplementedError


class PILlmdb(BaseLMDB):
    def __init__(
        self,
        lmdb_dir: Union[str, Path],
        image_list: Union[str, Path, pd.DataFrame]=None,
        index_key='id',
        **kwargs
    ):
        super().__init__(path=lmdb_dir, **kwargs)
        if image_list is None:
            self.ids = list(range(len(self.keys)))
            self.labels = list(range(len(self.ids)))
        else:
            df = pd.read_csv(str(image_list))
            assert index_key in df, f'[PILlmdb] Error! {image_list} must have id keys.'
            self.ids = df[index_key].tolist()
            assert max(self.ids) < len(self.keys)
            if 'label' in df:
                self.labels = df['label'].tolist()
            else:  # all numeric keys other than 'id' are labels
                keys = [key for key in df if (key!=index_key and type(df[key][0]) in [int, np.int64])]
                # df = df.drop('id', axis=1)
                self.labels = df[keys].to_numpy()
        self._length = len(self.ids)

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter([self.keys[i] for i in self.ids])

    def __getitem__(self, index):
        key = self.keys[self.ids[index]]
        return super().__getitem__(key)

    def set_ids(self, ids):
        self.ids = [self.ids[i] for i in ids]
        self.labels = [self.labels[i] for i in ids]
        self._length = len(self.ids)
        
    def _decode_value(self, value: bytes):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value))

    @staticmethod
    def write(indir, outdir, data_lst=None, transform=None):
        """
        create lmdb given data directory and list of image paths; or an iterator
        :param data_lst None or csv file containing 'path' key to store relative paths to the images
        :param indir root directory of the images
        :param outdir output lmdb, data.mdb and lock.mdb will be written here
        """
        
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path("/tmp") / f"TEMP_{time.time()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dtype = {'str': False, 'pil': False}
        if isinstance(indir, str) or isinstance(indir, Path):
            indir = Path(indir)
            if data_lst is None:  # grab all images in this dir
                lst = list(indir.glob('**/*.jpg')) + list(indir.glob('**/*.png'))
            else:
                lst = pd.read_csv(data_lst)['path'].tolist()
                lst = [indir/p for p in lst]
            assert len(lst) > 0, f'Couldnt find any image in {indir} (Support only .jpg and .png) or list (must have path field).'
            n = len(lst)
            dtype['str'] = True 
        else:  # iterator
            n = len(indir)
            lst = iter(indir)
            dtype['pil'] = True 

        with lmdb.open(path=str(tmp_dir), map_size=2 ** 40) as env:
            # Add the protocol to the database.
            with env.begin(write=True) as txn:
                key = "protocol".encode("ascii")
                value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
                txn.put(key=key, value=value, dupdata=False)
            # Add the keys to the database.
            with env.begin(write=True) as txn:
                key = pickle.dumps("keys")
                value = pickle.dumps(list(range(n)))
                txn.put(key=key, value=value, dupdata=False)
            # Add the images to the database.
            for key, value in tqdm(enumerate(lst), total=n, miniters=n//100, mininterval=300):
                with env.begin(write=True) as txn:
                    key = pickle.dumps(key)
                    if dtype['str']:
                        with value.open("rb") as file:
                            byteimg = file.read()
                    else:  # PIL
                        data = io.BytesIO()
                        value.save(data, 'png')
                        byteimg = data.getvalue()

                    if transform is not None:
                        im = Image.open(io.BytesIO(byteimg))
                        im = transform(im)
                        data = io.BytesIO()
                        im.save(data, 'png')
                        byteimg = data.getvalue()
                    txn.put(key=key, value=byteimg, dupdata=False)

        # Move the database to its destination.
        copy_tree(str(tmp_dir), str(outdir))
        shutil.rmtree(str(tmp_dir))



class MaskDatabase(PILlmdb):
    def _decode_value(self, value: bytes):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value)).convert("1")


class LabelDatabase(BaseLMDB):
    pass


class ArrayDatabase(BaseLMDB):
    _dtype = None
    _shape = None

    def __init__(
        self,
        lmdb_dir: Union[str, Path],
        image_list: Union[str, Path, pd.DataFrame]=None,
        **kwargs
    ):
        super().__init__(path=lmdb_dir, **kwargs)
        if image_list is None:
            self.ids = list(range(len(self.keys)))
            self.labels = list(range(len(self.ids)))
        else:
            df = pd.read_csv(str(image_list))
            assert 'id' in df, f'[ArrayDatabase] Error! {image_list} must have id keys.'
            self.ids = df['id'].tolist()
            assert max(self.ids) < len(self.keys)
            if 'label' in df:
                self.labels = df['label'].tolist()
            else:  # all numeric keys other than 'id' are labels
                keys = [key for key in df if (key!='id' and type(df[key][0]) in [int, np.int64])]
                # df = df.drop('id', axis=1)
                self.labels = df[keys].to_numpy()
        self._length = len(self.ids)

    def set_ids(self, ids):
        self.ids = [self.ids[i] for i in ids]
        self.labels = [self.labels[i] for i in ids]
        self._length = len(self.ids)

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter([self.keys[i] for i in self.ids])

    def __getitem__(self, index):
        key = self.keys[self.ids[index]]
        return super().__getitem__(key)

    @property
    def dtype(self):
        if self._dtype is None:
            protocol = self.protocol
            self._dtype = self._get(
                item="dtype",
                encode_key=lambda key: _default_encode(key, protocol=protocol),
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            protocol = self.protocol
            self._shape = self._get(
                item="shape",
                encode_key=lambda key: _default_encode(key, protocol=protocol),
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._shape

    def _decode_value(self, value: bytes) -> ndarray:
        value = super()._decode_value(value)
        return np.frombuffer(value, dtype=self.dtype).reshape(self.shape)

    def _decode_values(self, values: list) -> ndarray:
        shape = (len(values),) + self.shape
        return np.frombuffer(b"".join(values), dtype=self.dtype).reshape(shape)

    @staticmethod
    def write(diter, outdir):
        """
        diter is an iterator that has __len__ method
        class Myiter():
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                self.counter = 0
                return self
            def __len__(self):
                return len(self.data)
            def __next__(self):
                if self.counter < len(self):
                    out = self.data[self.counter]
                    self.counter+=1
                    return out
                else:
                    raise StopIteration
        a = iter(Myiter([1,2,3]))
        for i in a:
            print(i)
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path("/tmp") / f"TEMP_{time.time()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Create the database.
        n = len(diter)
        with lmdb.open(path=str(tmp_dir), map_size=2 ** 40) as env:
            # Add the protocol to the database.
            with env.begin(write=True) as txn:
                key = "protocol".encode("ascii")
                value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
                txn.put(key=key, value=value, dupdata=False)
            # Add the keys to the database.
            with env.begin(write=True) as txn:
                key = pickle.dumps("keys")
                value = pickle.dumps(list(range(n)))
                txn.put(key=key, value=value, dupdata=False)
            # Extract the shape and dtype of the values.
            value = next(iter(diter))
            shape = value.shape
            dtype = value.dtype
            # Add the shape to the database.
            with env.begin(write=True) as txn:
                key = pickle.dumps("shape")
                value = pickle.dumps(shape)
                txn.put(key=key, value=value, dupdata=False)
            # Add the dtype to the database.
            with env.begin(write=True) as txn:
                key = pickle.dumps("dtype")
                value = pickle.dumps(dtype)
                txn.put(key=key, value=value, dupdata=False)
            # Add the values to the database.
            with env.begin(write=True) as txn:
                for key, value in tqdm(enumerate(iter(diter)), total=n, miniters=n//100, mininterval=300):
                    key = pickle.dumps(key)
                    value = pickle.dumps(value)
                    txn.put(key=key, value=value, dupdata=False)

        # Move the database to its destination.
        copy_tree(str(tmp_dir), str(outdir))
        shutil.rmtree(str(tmp_dir))



class TensorDatabase(ArrayDatabase):
    def _decode_value(self, value: bytes) -> Tensor:
        return torch.from_numpy(super(TensorDatabase, self)._decode_value(value))

    def _decode_values(self, values: list) -> Tensor:
        return torch.from_numpy(super(TensorDatabase, self)._decode_values(values))
