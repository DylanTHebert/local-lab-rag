from vector_db.ingestion_source import ImageIngestionSource
from vector_db.ingestion_source import TextIngestionSource

if __name__ == "__main__":
    # create ingestion elements
    sources = [TextIngestionSource(), ImageIngestionSource()]
    # call service
    pass
