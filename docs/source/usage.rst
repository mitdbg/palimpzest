Usage
=====

.. _installation:

Installation
------------

To use Palimpzest, first install it using pip:

.. code-block:: console

   $ pip install palimpzest

.. Creating Dataset
.. ----------------
.. The ``Dataset`` object defines a node of computation in the semantic computation graph. The interface is defined below:

.. .. autofunction:: palimpzest.sets.Dataset

.. .. .. function:: Dataset(source: str | list | pd.DataFrame | DataSource, schema: Schema | None = None, *args, **kwargs)

.. ..    :param source: The source of data for the dataset. This can be a string, a list, a Pandas DataFrame, or a ``DataSource`` object.
.. ..    :param schema: The schema of the dataset. If not provided, it will be inferred from the source.


.. .. >>> [1, 2, 3]
.. .. [1, 2, 3]
