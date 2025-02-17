# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Optional, TypeVar


try:
    from tmetry.simpleevent import SimpleEventRecord
    from tmetry.writer import TmetryWriter

    b_tmetry_available = True
except ImportError:
    b_tmetry_available = False

VTYPE = TypeVar("T", str, int, bool, float)


class EventLogger:
    """
    Base class for providing event logging in a path handler
    It implements event logging by wrapping the tmetry interface.
    If tmetry packages is not available, it is a no-op.
    """

    DEFAULT_TOPIC = "iopath_tmetry"
    OP_KEY = "op"

    # Interval after which event is logged to backend.
    SAMPLING_PERIOD = 10

    # Map to keep track of sample count per operation.
    # pyre-fixme[4]: Attribute must be annotated.
    sample_counts = {}

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, *args, **kwargs):
        if b_tmetry_available:
            # pyre-fixme[4]: Attribute must be annotated.
            self._writers = []
            # pyre-fixme[4]: Attribute must be annotated.
            self._evt = SimpleEventRecord()
            # pyre-fixme[4]: Attribute must be annotated.
            self._enabled = True

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def add_writer(self, writer):
        if b_tmetry_available:
            if isinstance(writer, TmetryWriter):
                self._writers.append(writer)

    # pyre-fixme[3]: Return type must be annotated.
    def add_key(self, key: str, val: VTYPE):
        if b_tmetry_available:
            self._evt.set(key, val)

    # pyre-fixme[3]: Return type must be annotated.
    def add_keys(self, kvs: Dict[str, VTYPE]):
        if b_tmetry_available:
            self._evt.set_keys(kvs)

    def _sample_record(self) -> bool:
        """
        Samples the current event and logs only when the count
        reaches logging interval.

        Returns:
            True: if this sample should be logged.
            False: otherwise.
        """
        evt_op = self._evt.get(self.OP_KEY)
        if evt_op is None:
            # No op is set. Let's log it.
            return True

        if evt_op not in self.sample_counts:
            self.sample_counts[evt_op] = 1
            return True

        self.sample_counts[evt_op] += 1
        if self.sample_counts[evt_op] > self.SAMPLING_PERIOD:
            # Let's log this and reset sanpling counter.
            self.sample_counts[evt_op] = 1
            return True

        # Skip this sample.
        return False

    def set_logging(self, enable: bool) -> None:
        self._enabled = enable

    def is_logging_enabled(self) -> bool:
        return self._enabled

    # pyre-fixme[3]: Return type must be annotated.
    def log_event(self, topic: Optional[str] = None):
        if b_tmetry_available and self._enabled:

            # Sample the current event.
            if not self._sample_record():
                return

            if topic is None:
                topic = self.DEFAULT_TOPIC

            for writer in self._writers:
                writer.writeRecord(topic, self._evt)
            del self._evt
            self._evt = SimpleEventRecord()
