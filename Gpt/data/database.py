from __future__ import annotations

import copy
import ctypes
from collections.abc import Iterator

import lmdb
import torch


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor_size = tensor.numel() * tensor.element_size()
    tensor_ctype_ptr = ctypes.POINTER(ctypes.c_ubyte * tensor_size)
    return bytes(ctypes.cast(tensor.data_ptr(), tensor_ctype_ptr).contents)


class Database:
    def __init__(
        self,
        path: str,
        dtype: torch.dtype = torch.float32,
        readonly: bool = True,
        readahead: bool = False,
        meminit: bool = False,
        map_size: int = 1024 ** 4,
        **kwargs,
    ):
        self.path = path
        self.dtype = dtype
        self.readonly = readonly
        self.readahead = readahead
        self.meminit = meminit
        self.map_size = map_size
        kwargs["create"] = kwargs.get("create", not self.readonly)
        kwargs["lock"] = kwargs.get("lock", not self.readonly)
        self.kwargs = kwargs
        self._open_lmdb_env()

    def keys(self):
        with self.lmdb_env.begin() as txn:
            with txn.cursor() as cursor:
                keys = [key.decode() for key in cursor.iternext(values=False)]
        return keys

    def __contains__(self, key: str) -> bool:
        with self.lmdb_env.begin() as txn:
            value_bytes = txn.get(key.encode())
        contains = value_bytes is not None
        return contains

    def __getitem__(self, key: str) -> torch.Tensor:
        with self.lmdb_env.begin() as txn:
            value_bytes = txn.get(key.encode())
        if value_bytes is None:
            return KeyError(f"Key not in database: {key}")
        value = torch.frombuffer(bytearray(value_bytes), dtype=self.dtype)
        return value

    def __setitem__(self, key: str, value: torch.Tensor):
        value_bytes = tensor_to_bytes(value.type(self.dtype))
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key.encode(), value_bytes, dupdata=False)

    def set_bytes(self, key_bytes, value_bytes):
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key_bytes, value_bytes, dupdata=False)

    def __delitem__(self, key: str):
        with self.lmdb_env.begin(write=True) as txn:
            txn.delete(key.encode())

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for key in self.keys():
            value = self[key]
            yield key, value

    def __len__(self):
        return self.lmdb_env.stat()["entries"]

    def __del__(self):
        self.lmdb_env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lmdb_env"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._open_lmdb_env()

    def _open_lmdb_env(self):
        self.lmdb_env = lmdb.open(
            self.path,
            map_size=self.map_size,
            readonly=self.readonly,
            readahead=self.readahead,
            meminit=self.meminit,
            **self.kwargs,
        )
