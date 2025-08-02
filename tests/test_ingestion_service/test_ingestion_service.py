import tempfile
import shutil
import os
from pathlib import Path


def test_text_gather_samples():
    from vector_db.ingestion_source import TextIngestionSource

    ing = TextIngestionSource(r"tests\resources")
    assert len(ing) == 16


def test_ingestion():
    from vector_db.ingestion_service import IngestionService, TextIngestionSource
    import deeplake

    # Create temp directories for dataset and input
    with tempfile.TemporaryDirectory() as dataset_dir, tempfile.TemporaryDirectory() as input_dir:
        # Copy test txt files into input_dir
        test_files_src = Path("tests/resources/text")
        test_files = list(test_files_src.glob("*.txt"))
        for f in test_files:
            shutil.copy(f, input_dir)

        # Point DeepLake dataset at dataset_dir
        dataset = deeplake.dataset(dataset_dir)

        # Point ingestion service at input_dir
        source = TextIngestionSource(str(input_dir))
        service = IngestionService(dataset, [source])

        # Run ingestion (assuming service.run() processes all files and moves/deletes them)
        service.run()

        # Verify dataset length matches number of txt files
        assert len(dataset) == 16  # chunks them, should be 16

        # Verify no txt files remain in input_dir
        assert len(list(Path(input_dir).glob("*.txt"))) == 0
