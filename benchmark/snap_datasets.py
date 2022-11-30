import shutil
import gzip
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict
import requests
from pathlib import Path
import networkit as nk
from tqdm import tqdm
import csv
# from subprocess import Popen


class SnapGroundTruthDataset(ABC):
    def __init__(self, base_dir: Path, graph_url: str, labels_url: str) -> None:
        if not base_dir.is_dir():
            raise ValueError("{} is not a directory.".format(base_dir))
        self._base_dir = base_dir
        self._graph_url = graph_url
        self._labels_url = labels_url
        self._graph = None
        self._node_map = None

    def _download(self, url: str, typename: str) -> None:
        gz_file = self._base_dir / f"{self.name}_{typename}.txt.gz"
        txt_file = self._base_dir / f"{self.name}_{typename}.txt"

        with requests.get(url, stream=True) as r:
            total_size_in_bytes = int(
                r.headers.get('content-length', 0))
            r.raise_for_status()
            with open(gz_file, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=total_size_in_bytes/8192, desc=f"Downloading {self.name}_{typename}", unit_scale=True):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print("Unzipping ...")
        # Popen(
        #     f"gunzip {gz_file.resolve()}", shell=True).wait()
        with gzip.open(gz_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def _load_cover(self, url: str, typename: str = "cmty") -> nk.structures.Cover:
        path = self._base_dir / f"{self.name}_{typename}.txt"
        if not path.is_file():
            self._download(url, typename)
        # TODO: Use an integrated function for that
        cover = nk.Cover(n=self.graph.numberOfNodes())
        with open(path, 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter="\t")
            for i, row in enumerate(datareader):
                for j in row:
                    cover.addToSubset(i, self.node_map[j])
        return cover
        # return nk.graphio.CoverReader().read(str(path.resolve()), self.graph)
        # print(self._node_map)
        # return nk.graphio.SNAPEdgeListPartitionReader().read(str(path.resolve()), self._node_map, self.graph)
        # return nk.graphio.EdgeListPartitionReader(firstNode=0, sepChar=' ').read(
        #     str(path.resolve()))
        # return nk.graphio.EdgeListCoverReader(firstNode=0).read(str(path.resolve()), self.graph)

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        pass

    @property
    def graph(self) -> nk.Graph:
        if self._graph is None:
            path = self._base_dir / f"{self.name}_ungraph.txt"
            if not path.is_file():
                self._download(self._graph_url, typename="ungraph")
            reader = nk.graphio.EdgeListReader(
                '\t', 0, commentPrefix='#', continuous=False, directed=False)
            self._graph = reader.read(str(path.resolve()))
            self._node_map = reader.getNodeMap()
        return self._graph

    @property
    def node_map(self) -> Dict[str, int]:
        if self._node_map is None:
            self.graph
        return self._node_map

    @property
    def labels(self) -> nk.structures.Cover:
        return self._load_cover(self._labels_url)


class ComDataset(SnapGroundTruthDataset, metaclass=ABCMeta):
    def __init__(self, base_dir: Path) -> None:
        graph_url = f"https://snap.stanford.edu/data/bigdata/communities/com-{self.name}.ungraph.txt.gz"
        labels_url = f"https://snap.stanford.edu/data/bigdata/communities/com-{self.name}.all.cmty.txt.gz"
        self._top_labels_url = f"https://snap.stanford.edu/data/bigdata/communities/com-{self.name}.top5000.cmty.txt.gz"
        super().__init__(base_dir, graph_url=graph_url, labels_url=labels_url)

    @property
    def labels_top5000(self) -> nk.structures.Cover:
        return self._load_cover(self._top_labels_url, "top5000_cmty")


class EmailCoreDataset(SnapGroundTruthDataset):
    def __init__(self, base_dir: Path) -> None:
        graph_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
        labels_url = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
        super().__init__(base_dir, graph_url=graph_url, labels_url=labels_url)

    @property
    def labels(self) -> nk.structures.Cover:
        path = self._base_dir / f"{self.name}_cmty.txt"
        if not path.is_file():
            self._download(self._labels_url, "cmty")
        # return nk.graphio.EdgeListPartitionReader(firstNode=0, sepChar=' ').read(
        #     str(path.resolve()))
        return nk.graphio.EdgeListCoverReader(firstNode=0).read(str(path.resolve()), self.graph)

    @classmethod
    @property
    def name(cls) -> str:
        return 'email'


class WikiDataset(SnapGroundTruthDataset):
    def __init__(self, base_dir: Path) -> None:
        graph_url = "https://snap.stanford.edu/data/wiki-topcats.txt.gz"
        labels_url = "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz"
        super().__init__(base_dir, graph_url=graph_url, labels_url=labels_url)

    @property
    def labels(self) -> nk.structures.Cover:
        path = self._base_dir / f"{self.name}_cmty.txt"
        if not path.is_file():
            self._download(self._labels_url, "cmty")
        cover = nk.Cover(n=self.graph.numberOfNodes())
        with open(path, 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=" ")
            for i, row in enumerate(datareader):
                if row[1] != '':
                    # If category is not empty
                    for j in row[1:]:
                        cover.addToSubset(i, self.node_map[j])
        return cover

    @classmethod
    @property
    def name(cls) -> str:
        return 'wiki'


class LiveJournalDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'lj'


class FriendsterDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'friendster'


class OrkutDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'orkut'


class YoutubeDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'youtube'


class DblpDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'dblp'


class AmazonDataset(ComDataset):
    @classmethod
    @property
    def name(cls) -> str:
        return 'amazon'


class TestDataset(SnapGroundTruthDataset):
    def __init__(self, base_dir: Path) -> None:
        self._graph = nk.Graph(5)
        self._graph.addEdge(0, 1)
        self._graph.addEdge(1, 2)
        self._graph.addEdge(0, 2)
        self._graph.addEdge(3, 4)
        self._node_map = None

    @property
    def graph(self) -> nk.Graph:
        return self._graph

    @property
    def labels(self) -> nk.structures.Cover:
        cc = nk.components.ConnectedComponents(self._graph)
        cc.run()
        return nk.structures.Cover(cc.getPartition())

    @classmethod
    @property
    def name(cls) -> str:
        return 'test'
