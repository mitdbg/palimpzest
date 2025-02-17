# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime as dt
import io
import logging
import os
import shutil
import types
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, IO, List, Optional, Tuple, Union

from iopath.common.file_io import file_lock, get_cache_dir, PathHandler


try:
    # Needed for S3 PathHandler
    # pyre-fixme[21]: Could not find module `boto3`.
    import boto3

    # pyre-fixme[21]: Could not find module `botocore`.
    import botocore
except ImportError:
    boto3 = None
    botocore = None


# Override for close() on files to write to Amazon S3
# pyre-fixme[2]: Parameter must be annotated.
def s3_close_and_upload(self, client, bucket, s3_path, transfer_config) -> None:
    # Seek to start, for use by upload_fileobj.
    self.seek(0)

    # Reinstall the proper close.
    self.close = self._close

    # upload_fileobj needs bytes
    # NOTE: This is undesirable.
    if isinstance(self, io.StringIO):
        self = io.BytesIO(self.getvalue().encode("utf-8"))

    # Upload
    try:
        client.upload_fileobj(
            self,
            bucket,
            s3_path,
            Config=transfer_config,
        )
    except botocore.exceptions.ClientError as e:
        raise OSError(f"Error in file upload - {e}" f"{type(e).__name__}: {e}") from e


class S3PathHandler(PathHandler):
    """
    Support for Amazon Simple Storage Service (S3)

    PathHanlder methods, at a glance:

     File     --torch.load->     In     --open(..., 'w')->   Amazon    <- _exists,_isfile,_isdir,_ls,_rm ...
    System   <-torch.save--     Mem.   <-open(..., 'r')--      S3
            <----------------_copy_from_local-----------------
            ----------------_get_local_path ----------------->

    Mem usage, for processing N bytes:
        open(..., mode)
            mode=='w':    2N,  due to fully buffering user input,
                                *and doing naive conversion from StringIO -> BytesIO*,
                                before writing to S3
                                ^ Potential for optimization.
            mode=='wb':    N,  due to fully buffering user input, before writing to S3.
            mode=='r':     N,  due to fully buffering file in memory
            mode=='rb':    N,  due to fully buffering file in memory
        _copy_from_local: ≈0.  boto3 streams from file system directly to s3
        _get_local_path:  ≈0.  boto3 streams from s3 directly from s3 to file system

    NOTE:
        S3 doesn't have a notion of directories.  This pathhandler simulates
        directories via uploading objects with a name ending in a slash, on calls to mkdir().
        ls() calls return objects as if they were in a directory structure, via
        boto3's options.
    """

    # Disable failures if not all args are specified.
    _strict_kwargs_check = False

    S3_PREFIX = "s3://"
    CACHE_SUBDIR_NAME = "s3_cache"

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        profile: Optional[str] = "saml",
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        transfer_config_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            cache_dir (str): Local filesystem directory to use for caching. If None,
                uses default from `file_io.get_cache_dir()`.
            transfer_config_kwargs (dict): Settings for boto3.s3.transfer.TransferConfig.
                Used to specify settings for multipart transfers.
                See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html for details.
        """
        self.cache_dir = cache_dir
        self.profile = profile
        # pyre-fixme[21]: Could not find module `boto3.s3.transfer`.
        from boto3.s3.transfer import TransferConfig

        # pyre-fixme[4]: Attribute must be annotated.
        self.transfer_config = TransferConfig(
            **(transfer_config_kwargs if transfer_config_kwargs else {})
        )

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        return [self.S3_PREFIX]

    def _parse_uri(self, uri: str) -> Tuple[str, str]:
        """
        Parses a "s3://bucket/path" URI into `bucket` and `path` strings.

        Args:
            uri (str): A s3:// URI.

        Returns:
            bucket (str): the s3 bucket.
            path (str): the path on the s3 system.
        """
        splits = uri.replace(self.S3_PREFIX, "").split("/")
        bucket = splits[0]
        path = "/".join(splits[1:])
        return bucket, path

    # pyre-fixme[3]: Return type must be annotated.
    def _get_client(self, bucket: str):
        logger = logging.getLogger(__name__)
        if not hasattr(self, "client"):
            try:
                session = boto3.Session(profile_name=self.profile)
                # pyre-fixme[16]: `S3PathHandler` has no attribute `client`.
                self.client = session.client("s3")
            except botocore.exceptions.NoCredentialsError as e:
                logger.error(
                    " See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html "
                    " for method of using environment variable to point to aws credentials, and the "
                    " order in which boto will search for said credentials. "
                )
                logger.error(
                    "Boto3 searches via the order below.  If on FAIR Cluster, method 4 may be most convenient."
                    ""
                    "The order in which Boto3 searches for credentials is:"
                    "1) [UNUSED] Passing credentials as parameters in the boto.client() method"
                    "2) [UNUSED] Passing credentials as parameters when creating a Session object"
                    "3) Environment variables"
                    "       AWS_ACCESS_KEY_ID - The access key for your AWS account."
                    "       AWS_SECRET_ACCESS_KEY - The secret key for your AWS account."
                    "       AWS_SESSION_TOKEN - The session key for your AWS account."
                    "           This is only needed when you are using temporary credentials. "
                    "4) Shared credential file (~/.aws/credentials)"
                    "       default: ~/.aws/credentials"
                    "       changed via: AWS_SHARED_CREDENTIALS_FILE"
                    "       *for FAIR cluster usage: `export AWS_SHARED_CREDENTIALS_FILE=~/.fairusers_aws/credentials`"
                    "5) AWS config file (~/.aws/config)"
                    "       default: ~/.aws/config"
                    "       changed via: AWS_CONFIG_FILE"
                    "6) Assume Role provider"
                    "7) Boto2 config file (/etc/boto.cfg and ~/.boto)"
                    "8) Instance metadata service on an Amazon EC2 instance that has an IAM role configured."
                )
                raise OSError(
                    f"Error in making s3 client for bucket {bucket}"
                    f"{type(e).__name__}: {e}"
                ) from e

        return self.client

    # pyre-fixme[3]: Return type must be annotated.
    def _local_cache_path(
        self,
        path: str,
    ):
        """
        Helper that returns a local cache path for a given uri.
        Args:
            path (str): A URI supported by this PathHandler.
        Returns:
            local_cache_path (str): a file path which exists on the local file system,
            in a cache directory.
        """
        bucket, file_path = self._parse_uri(path)
        return os.path.join(
            get_cache_dir(self.cache_dir), self.CACHE_SUBDIR_NAME, file_path
        )

    # pyre-fixme[14]: `_get_local_path` overrides method defined in `PathHandler`
    #  inconsistently.
    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)

        # Cheap check first.
        if path.endswith("/"):
            raise NotImplementedError(
                "S3PathHandler does not currently support downloading directories"
            )
        assert self._isfile(path)

        local_path = self._local_cache_path(path)
        with file_lock(local_path):
            if os.path.exists(local_path):
                # If local object's last modified time is *after* remote object's last modified
                # time, do not use the cache.  Instead, redownload.
                response = self._head_object(path)
                if response is not None:
                    remote_dt = response["LastModified"]
                    local_dt = dt.datetime.fromtimestamp(
                        os.path.getmtime(local_path)
                    ).astimezone()
                    # NOTE: may consider still avoid cache if times are close, to avoid a race condition.
                    # Currently, a lengthy download of a very recent but stale file would have a late
                    # local last modified timestamp, and would be improperly used.
                    # Better fix: set last modified time via the remote object's last modified time,
                    # in download_file().
                    # pyre-fixme[58]: `>` is not supported for operand types
                    #  `datetime` and `timedelta`.
                    if (local_dt - remote_dt) > dt.timedelta(minutes=0):
                        logger.info(
                            "URL {} was already cached in {}".format(path, local_path)
                        )
                        return local_path

            logger.info("Caching {} ...".format(path))
            tmp = local_path + ".tmp"
            # clean-up tmp if found, because if tmp exists, it must be a dirty
            # result of a previously process that didn't cleanup itself.
            if os.path.isfile(tmp):
                os.unlink(tmp)

            bucket, s3_path = self._parse_uri(path)
            client = self._get_client(bucket)
            try:
                response = client.download_file(
                    bucket, s3_path, tmp, Config=self.transfer_config
                )

                # First download to tmp, then move it, because move is
                # (almost?) atomic when src and dst are in the same file
                # system. This will avoid partial cache state if the
                # process is killed.
                shutil.move(tmp, local_path)
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

            logger.info("URL {} cached in {}".format(path, local_path))
            return local_path

    # pyre-fixme[15]: `_copy_from_local` overrides method defined in `PathHandler`
    #  inconsistently.
    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a local file to the specified URI.
        If the URI is another local path, this should be functionally identical
        to copy.
        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI
        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        # Just checking this to avoid expensive API calls in self._isdir().
        if local_path.endswith("/") or dst_path.endswith("/"):
            raise NotImplementedError(
                "S3PathHandler does not currently support uploading directories"
            )

        if not overwrite and self._exists(dst_path):
            logger = logging.getLogger(__name__)
            logger.error("Error: Destination path {} already exists.".format(dst_path))
            return False

        bucket, s3_path = self._parse_uri(dst_path)
        client = self._get_client(bucket)
        try:
            client.upload_file(local_path, bucket, s3_path, Config=self.transfer_config)
            return True
        except botocore.exceptions.ClientError as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file upload - {}".format(str(e)))
            return False

    # pyre-fixme[3]: Return type must be annotated.
    def _decorate_buf_with_s3_methods(
        self,
        buffer: Union[IO[str], IO[bytes]],
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Any,
        bucket: str,
        s3_path: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        transfer_config: Any,
    ):
        # Save old close method.
        # pyre-fixme[16]: `IO` has no attribute `_close`.
        buffer._close = buffer.close

        # Add in our new close method.
        fn = partial(
            s3_close_and_upload,
            client=client,
            bucket=bucket,
            s3_path=s3_path,
            transfer_config=transfer_config,
        )
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[typing.Callable(IO.close)[[Named(self, IO[bytes])], None],
        #  IO[bytes]]`; used as `MethodType`.
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[typing.Callable(IO.close)[[Named(self, IO[str])], None],
        #  IO[str]]`; used as `MethodType`.
        # pyre-fixme[9]: close has type
        #  `Union[BoundMethod[typing.Callable(IO.close)[[Named(self, IO[bytes])],
        #  None], IO[bytes]], BoundMethod[typing.Callable(IO.close)[[Named(self,
        #  IO[str])], None], IO[str]]]`; used as `MethodType`.
        buffer.close = types.MethodType(fn, buffer)

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        # The following three arguments are unused,
        # But are included to avoid triggering WARNING
        # messages from _check_kargs.
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        read_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.
        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)

        bucket, s3_path = self._parse_uri(path)
        client = self._get_client(bucket)

        # AWS methods download_fileobj() and upload_fileobj()
        # both expect binary file-like objects.
        if "r" in mode:
            if read_chunk_size is None:
                # 1. Download into io.BytesIO.
                # (binary format is required by download_fileobj.)
                buffer = io.BytesIO()
                try:
                    # NOTE: Will download entire file!  Further optimization to
                    # only read a portion of the file could be implemented here.
                    # NOTE: We download into an in-memory buffer.  If downloading to
                    # filesystem is desirable, use _get_local_path().
                    client.download_fileobj(
                        bucket, s3_path, buffer, Config=self.transfer_config
                    )
                except botocore.exceptions.ClientError as e:
                    raise OSError(
                        f"Error in making s3 client for bucekt {bucket}"
                        f"{type(e).__name__}: {e}"
                    ) from e

                # 2. Set file-pointer to beginning of file.
                buffer.seek(0)
            else:
                # pyre-fixme[6]: For 3rd param expected `int` but got `str`.
                buffer = S3ChunkReadIO(client, bucket, s3_path, read_chunk_size)
            # pyre-fixme[16]: `S3PathHandler` has no attribute `length`.
            self.length = client.get_object(Bucket=bucket, Key=s3_path)["ContentLength"]

            # 3. Use convenient wrapper to make object look like StringIO,
            # if user wants non-binary.
            encoding = None

            if "b" not in mode:
                encoding = "utf-8"
                return io.TextIOWrapper(
                    # pyre-fixme[6]: For 1st param expected `IO[bytes]` but got
                    #  `Union[S3ChunkReadIO, BytesIO]`.
                    buffer,
                    write_through=True,
                    encoding=encoding,
                    errors=None,
                    newline=None,
                    line_buffering=False,
                )
            else:
                # pyre-fixme[7]: Expected `Union[IO[bytes], IO[str]]` but got
                #  `Union[S3ChunkReadIO, BytesIO]`.
                return buffer

        elif "w" in mode:
            # 1. For writing, we give the user io.BytesIO or io.StringIO.
            if "b" in mode:
                buffer = io.BytesIO()
            else:
                buffer = io.StringIO()

            # 2. Decorate buffer so that we upload when it's closed by user.
            #       If StringIO, decorator does a simple+expensive conversion
            #       to bytesIO before uploading.
            #       (because upload_fileobj requires binary)
            self._decorate_buf_with_s3_methods(
                buffer, client, bucket, s3_path, self.transfer_config
            )

            return buffer

        else:
            raise OSError(f"Unsupported open mode {mode}")

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        if not overwrite and self._exists(dst_path):
            logger = logging.getLogger(__name__)
            logger.error("Error: Destination path {} already exists.".format(dst_path))
            return False

        src_bucket, src_s3_path = self._parse_uri(src_path)
        dst_bucket, dst_s3_path = self._parse_uri(dst_path)
        assert src_bucket == dst_bucket, "For now, can only _copy() within a bucket."
        client = self._get_client(src_bucket)

        try:
            client.copy(
                {
                    "Bucket": src_bucket,
                    "Key": src_s3_path,
                },
                dst_bucket,
                dst_s3_path,
                Config=self.transfer_config,
            )
            return True
        except botocore.exceptions.ClientError as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict` to avoid runtime subscripting errors.
    def _head_object(self, path: str) -> Optional[Dict]:
        bucket, s3_path = self._parse_uri(path)
        client = self._get_client(bucket)

        try:
            # Raises exception if not exists, else it exists.
            response = client.head_object(Bucket=bucket, Key=s3_path)
            return response
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Message"] == "Bad Request":
                raise OSError(
                    f"Error in checking s3 path {path} - " f"{type(e).__name__}: {e}"
                ) from e
            return None

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path exists
        """
        self._check_kwargs(kwargs)

        return self._head_object(path) is not None

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a file
        """
        self._check_kwargs(kwargs)

        # NOTE: this incurs an API call.
        return not path.endswith("/") and self._exists(path, **kwargs)

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a directory
        """
        self._check_kwargs(kwargs)

        # NOTE: this incurs an API call.
        return path.endswith("/") and self._exists(path, **kwargs)

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            List[str]: list of contents in given path
        """
        self._check_kwargs(kwargs)

        bucket, s3_path = self._parse_uri(path)
        client = self._get_client(bucket)

        try:
            # Pagination needed if >1000 entries.
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket,
                Prefix=s3_path,
                Delimiter="/",
            )
            obj_results = [
                obj["Key"] for page in pages for obj in page.get("Contents", [])
            ]
            dir_results = [
                obj["Prefix"]
                for page in pages
                for obj in page.get("CommonPrefixes", [])
            ]
            return obj_results + dir_results
        except botocore.exceptions.ClientError as e:
            raise OSError(
                f"Error in ls path {path} - " f"{type(e).__name__}: {e}"
            ) from e

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.
        Args:
            path (str): A URI supported by this PathHandler
        """
        self._check_kwargs(kwargs)

        assert path.endswith("/"), path

        bucket, s3_path = self._parse_uri(path)
        client = self._get_client(bucket)

        try:
            client.put_object(Bucket=bucket, Key=s3_path)
        except botocore.exceptions.ClientError as e:
            raise OSError(
                f"Error in mkdirs path {path} - " f"{type(e).__name__}: {e}"
            ) from e

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        """
        self._check_kwargs(kwargs)

        bucket, s3_path = self._parse_uri(path)
        client = self._get_client(bucket)

        try:
            client.delete_object(Bucket=bucket, Key=s3_path)
        except botocore.exceptions.ClientError as e:
            raise OSError(
                f"Error in rm path {path} - " f"{type(e).__name__}: {e}"
            ) from e


class S3ChunkReadIO(io.BufferedIOBase):
    # pyre-fixme[4]: Attribute must be annotated.
    DEFAULT_CHUNK_SIZE = 50 * 1024 * 1024  # 50MB

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        client,
        bucket: str,
        key: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: Optional[timedelta] = None,
    ):
        # pyre-fixme[4]: Attribute must be annotated.
        self.client = client
        self.bucket = bucket
        self.key = key
        # pyre-fixme[4]: Attribute must be annotated.
        self.timeout = timeout.total_seconds() if timeout is not None else None
        self.chunk_size = chunk_size
        self.offset = 0
        self.buffered_window = range(0, 0)
        self.buffer = io.BytesIO()
        # pyre-fixme[4]: Attribute must be annotated.
        self.length = client.get_object(Bucket=bucket, Key=key)["ContentLength"]

    @property
    def name(self) -> str:
        # pyre-fixme[16]: `S3ChunkReadIO` has no attribute `path`.
        return self.path

    def seekable(self) -> bool:
        """
        Return a bool indicating whether object supports random access.

        If False, seek(), tell() and truncate() will raise OSError.
        This method may need to do a test seek().
        """
        return True

    def readable(self) -> bool:
        """
        Return a bool indicating whether object was opened for reading.

        If False, read() will raise OSError.
        """
        return True

    def writable(self) -> bool:
        """
        Return a bool indicating whether object was opened for writing.

        If False, write() and truncate() will raise OSError.
        """
        return False

    def fileno(self) -> int:
        raise AttributeError()

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Change stream position.

        Change the stream position to byte offset offset. Argument offset is
        interpreted relative to the position indicated by whence.  Values
        for whence are ints:

        * 0 -- start of stream (the default); offset should be zero or positive
        * 1 -- current stream position; offset may be negative
        * 2 -- end of stream; offset is usually negative
        Some operating systems / file systems could provide additional values.

        Return an int indicating the new absolute position.
        """
        if whence == 0:
            assert offset >= 0
            self.offset = offset
        elif whence == 1:
            assert offset + self.offset >= 0
            self.offset += offset
        elif whence == 2:
            self.offset = self.length + offset
        return self.offset

    def tell(self) -> int:
        """Return an int indicating the current stream position."""
        return self.offset

    def truncate(self, size: Optional[int] = None) -> int:
        """
        Truncate file to size bytes.

        Size defaults to the current IO position as reported by tell().  Return

        the new size.
        """
        raise OSError("can't truncate readonly stream")

    # pyre-fixme[14]: `write` overrides method defined in `BufferedIOBase`
    #  inconsistently.
    # pyre-fixme[15]: `write` overrides method defined in `BufferedIOBase`
    #  inconsistently.
    def write(self, b: Union[bytes, bytearray]) -> Optional[int]:
        """
        Write bytes b to in-memory buffer, return number written.
        """
        raise OSError("can't write to readonly stream")

    def close(self) -> None:
        """
        noop
        """
        pass

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    # pyre-fixme[14]: `read` overrides method defined in `BufferedIOBase`
    #  inconsistently.
    def read(self, size: int = -1) -> bytes:
        """
        Read and return up to size bytes. If the argument is omitted, None, or negative,
        data is read and returned until EOF is reached. An empty bytes object is
        returned if the stream is already at EOF.
        """

        if size is None or size < 0:
            size = self.length - self.offset

        size = min(size, self.length - self.offset)

        ret = bytearray()

        if self.offset in self.buffered_window:
            buffer_offset = self.offset - self.buffered_window.start
            ret += self.buffer.getbuffer()[
                buffer_offset : min(buffer_offset + size, len(self.buffered_window))
            ]

        # if we already get enough data, return
        if len(ret) == size:
            self.offset += len(ret)
            return bytes(ret)

        # if partial data is available in the buffer, get the remaining data from S3
        if size - len(ret) > self.chunk_size:
            self.offset += len(ret)
            # For s3, range x-x means 1 byte at offset x
            output = self._read_from_s3(
                range(self.offset, min(self.offset + size - len(ret) - 1, self.length))
            )
            self.offset += len(output)
            # pyre-fixme[7]: Expected `bytes` but got `bytearray`.
            return ret + output

        # otherwise download the next chunk from s3, update buffer and buffered window
        self._read_chunk_to_buffer(self.offset + len(ret))

        # append the remaining data from newly downloaded buffer and return
        ret += self.buffer.getbuffer()[0 : size - len(ret)]

        assert len(ret) == size
        self.offset += len(ret)
        return bytes(ret)

    def _read_from_s3(self, download_range: range) -> bytes:
        obj = self.client.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={download_range.start}-{download_range.stop}",
        )
        streaming_body = obj["Body"]

        if self.timeout is not None:
            streaming_body.set_socket_timeout(self.timeout)

        ret = bytearray()
        for chunk in streaming_body.iter_chunks(chunk_size=self.chunk_size):
            ret += chunk
        streaming_body.close()
        # pyre-fixme[7]: Expected `bytes` but got `bytearray`.
        return ret

    def _read_chunk_to_buffer(self, start_offset: int) -> None:
        """
        download a chuck size of data start from start_offset into current buffer, then update
        self.buffered_window for booking which part of data is currently buffered
        """
        download_range = range(
            start_offset, min(start_offset + self.chunk_size, self.length)
        )

        ret = self._read_from_s3(download_range)

        self.buffer.seek(0)
        self.buffer.write(ret)
        self.buffered_window = download_range
