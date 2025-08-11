from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Any, Dict, List, Optional
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode

from homelab_rag.models.embeddings import FlagEmbedding, LocalEmbedding


class DataIngestionSource(ABC):
    def __init__(
        self, source_dir: Optional[str] = None, embedding: LocalEmbedding | None = None
    ) -> None:
        super().__init__()
        self.samples: List[TextNode] = []
        if source_dir is None:
            raise ValueError("No source path specified")
        else:
            self.source_dir = Path(source_dir)

    @abstractmethod
    def has_next(self) -> bool:
        """True if the source has another example to yield"""
        pass

    @abstractmethod
    def grab_sample(self) -> Dict[str, Any]:
        """Any length dict, where each string is a valid type in a deeplake store"""
        pass

    @abstractmethod
    def gather_samples(self) -> bool:
        """Any length dict, where each string is a valid type in a deeplake store"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def remove_candidates(self, items: set):
        """set of items that can be removed if there are no more instances of it in the samples

        Args:
            items: candidate items for removal
        """
        pass


class TextIngestionSource(DataIngestionSource):
    def __init__(
        self,
        source_dir: Optional[str] = None,
        embedding: LocalEmbedding = FlagEmbedding(),
    ) -> None:
        source_dir = (
            r"F:\data\datastores\drop_points\text" if source_dir is None else source_dir
        )
        super().__init__(source_dir)
        self.embedder = embedding
        self.gather_samples()

    def has_next(self) -> bool:
        if len(self.samples) > 0:
            return True
        return False

    def grab_sample(self) -> Dict[str, Any]:
        if len(self.samples) > 0:
            samp = self.samples.pop()
            prep = {
                "text": samp.text,
                "start_idx": samp.start_char_idx,
                "end_idx": samp.end_char_idx,
                "embedding": self.embedder(samp.text),
                "filename": samp.metadata["filename"],
                "source_path": samp.metadata["source_path"],
            }
            return prep
        else:
            raise IndexError()

    def gather_samples(self) -> bool:
        queried = self.source_dir.rglob("*.txt")
        docs = []
        fpaths = []
        for q in queried:
            fpaths.append(os.path.join(self.source_dir, q))
            doc = FlatReader().load_data(q)
            # Optionally, attach metadata to each doc before splitting
            for d in doc:
                d.metadata = {"filename": q.name, "source_path": str(q)}
            docs.extend(doc)
        if len(docs) > 0:
            splitter = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=20,
                separator=" ",
            )
            nodes = splitter.get_nodes_from_documents(docs)
            self.samples.extend(nodes)
        return len(self.samples) > 0

    def remove_candidates(self, items: set):
        """set of items that can be removed if there are no more instances of it in the samples

        Args:
            items: candidate items for removal
        """
        items_in_samp = set([i.metadata["source_path"] for i in self.samples])
        diff_items = items.difference(items_in_samp)
        for i in diff_items:
            if os.path.exists(i):
                os.remove(i)

    def __len__(self) -> int:
        return len(self.samples)


class ImageIngestionSource(DataIngestionSource):
    def __init__(
        self, source_dir: Optional[str] = None, embedding: LocalEmbedding = None
    ) -> None:
        source_dir = (
            r"F:\data\datastores\drop_points\images"
            if source_dir is None
            else source_dir
        )
        super().__init__(source_dir)
        raise NotImplementedError()

    def has_next(self) -> bool:
        # TODO: Implement logic to check for next image sample
        return False

    def grab_sample(self) -> Dict[str, Any]:
        # TODO: Implement logic to grab an image sample
        return {}

    def gather_samples(self) -> bool:
        return len(self.samples) > 0

    def __len__(self) -> int:
        return len(self.samples)
