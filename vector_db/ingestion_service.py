from collections import defaultdict
import sys
import time
import deeplake
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from vector_db.ingestion_source import (
    DataIngestionSource,
    ImageIngestionSource,
    TextIngestionSource,
)


class IngestionService:
    """Persistent service which ingest files from given endpoints and stores them in a local DeepLake dataset"""

    def __init__(
        self,
        dataset: deeplake.Dataset,
        ingestion_sources: List[DataIngestionSource],
        batch_size: int = 100,
    ) -> None:
        self.ds = dataset
        self.ingesting = False
        self.ingestion_sources = ingestion_sources
        self.batch_size = batch_size

    def run(self):
        self.ingesting = False
        while True:
            if self.stop:
                break
            if all([not i.has_next() for i in self.ingestion_sources]):
                self.is_empty = all(
                    [not i.gather_samples() for i in self.ingestion_sources]
                )
            for source in self.ingestion_sources:
                if source.has_next():
                    batch = self.batcher(source)
                    removed = set([i["source_path"] for i in batch])
                    self.ingest(batch)
                    source.remove_candidates(removed)
        time.sleep(10)

    def batcher(self, source: DataIngestionSource):
        avail_samps = len(source)
        if avail_samps > self.batch_size:
            samps_to_grab = self.batch_size
        else:
            samps_to_grab = avail_samps
        batch = defaultdict(list)
        for i in range(samps_to_grab):
            samp = source.grab_sample()
            for i, j in samp.items():
                batch[i].append(j)
        return batch

    def ingest(self, data: Dict[str, Any]) -> None:
        """This iters keys and adds them to the datastore"""
        # TODO this is where i am
        pass

    def stop(self):
        self._stop = True


def main():
    """
    Entry point for the ingestion service.
    """
    if len(sys.argv) < 2:
        print("Usage: python ingestion_service.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    Path.mkdir(Path(dataset_path), exist_ok=True)
    print(f"Starting ingestion service with dataset path: {dataset_path}")
    dataset: deeplake.Dataset = deeplake.dataset(dataset_path)
    service = IngestionService(dataset, [TextIngestionSource()])
    service.run()


if __name__ == "__main__":
    main()
