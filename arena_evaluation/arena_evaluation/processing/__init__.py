"""The L1 processing layer: extraction, alignment, dimensions and the evaluation pipeline.

The reader and the pipeline are imported from their own modules. They are not re-exported here:
the storage schema consumers import the metric registry, and an eager pipeline import would close
a contract -> layout -> registry -> pipeline -> contract import cycle.
"""
