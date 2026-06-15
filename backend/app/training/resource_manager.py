"""
Resource Manager — adjusts training resource allocation based on dataset size.

Determines parallelism and batch sizing so small datasets don't waste
resources and large datasets get enough throughput.
"""

import logging
from typing import Any

logger = logging.getLogger("resource_manager")


def adjust_training_resources(dataset_size: int) -> dict[str, Any]:
    """
    Recommend resource settings based on *dataset_size* (number of rows).

    Parameters
    ----------
    dataset_size : int
        Number of rows in the training dataset.

    Returns
    -------
    dict with ``parallelism`` and ``batch_size``.
    """
    if dataset_size < 50_000:
        profile = "small"
        parallelism = 1
        batch_size = None
    elif dataset_size < 200_000:
        profile = "medium"
        parallelism = 2
        batch_size = 10_000
    else:
        profile = "large"
        parallelism = 4
        batch_size = 25_000

    result = {
        "profile": profile,
        "parallelism": parallelism,
        "batch_size": batch_size,
        "dataset_size": dataset_size,
    }

    logger.info(
        "Resource profile: %s (rows=%d, parallelism=%d, batch_size=%s)",
        profile, dataset_size, parallelism, batch_size,
    )
    return result
