from palimpzest.core.data.dataset import Dataset


class Context(Dataset):
    """
    A `Context` is a special type of `Dataset` that represents a view over an underlying `Dataset`.
    Each `Context` has a `name` which uniquely identifies it, as well as a natural language `description`
    of the data / computation that the `Context` represents. Similar to `Dataset`s, `Context`s can be
    lazily transformed using functions such as `sem_filter`, `sem_map`, `sem_join`, etc., and they may
    be materialized or unmaterialized.
    """

    def __init__(self):
        super().__init__(source=source, schema=schema)