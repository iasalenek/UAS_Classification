import pandas as pd
from treelib import Tree
from typing_extensions import Self


class TaxonomyNode:
    def __init__(self,
                 name: str,
                 parent: Self | None = None,
                 textsDataFrame: pd.DataFrame | None = None,
                 ) -> None:
        self.name = name
        self.parent = parent
        self.children: dict[str, Self] = dict()
        self.textsDataFrame = textsDataFrame
        if self.textsDataFrame is not None:
            self.examplesLoc: dict[int, list] = {
                _cls: [] for _cls in self.textsDataFrame['Cls'].unique()}

    def addChild(self, child: Self) -> None:
        self.children[child.name] = child

    def getChild(self, child: str) -> Self:
        return self.children[child]

    def __getitem__(self, name: str) -> Self:
        return self.getChild(name)

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def getLeafs(self) -> dict[str, Self]:
        leafs = dict()
        for childName, child in self.children.items():
            if child.isLeaf():
                leafs[childName] = child
            else:
                leafs |= child.getLeafs()
        return leafs

    def _validateExamplesLoc(self, examplesLoc: list[int], examplesCls: int) -> None:
        assert (self.textsDataFrame.loc[examplesLoc]
                ['Cls'] == examplesCls).all()
        if self.isLeaf():
            assert (
                self.textsDataFrame.loc[examplesLoc]['Label'] == self.name).all()
        else:
            assert (self.textsDataFrame.loc[examplesLoc]['Label'].isin(
                self.getLeafs().keys())).all()

    def addExamplesLoc(self, _cls: int, examplesLoc: list[int]) -> None:
        self._validateExamplesLoc(examplesLoc, _cls)
        self.examplesLoc[_cls] += examplesLoc

    def setExamplesLoc(self) -> tuple[list, list]:

        if self.isLeaf() and (self.textsDataFrame is not None):
            for _cls in self.examplesLoc.keys():
                examplesLoc = self.textsDataFrame[(
                    self.textsDataFrame['Label'] == self.name) * (self.textsDataFrame['Cls'] == _cls)].index.tolist()
                self.addExamplesLoc(_cls, examplesLoc)
            return self.examplesLoc

        elif self.textsDataFrame is not None:
            for child in self.children.values():
                examplesLoc = child.setExamplesLoc()
                for _cls in self.examplesLoc.keys():
                    self.examplesLoc[_cls] += examplesLoc[_cls]
            return self.examplesLoc

        return self.examplesLoc

    def getExamples(self, _cls: int) -> pd.DataFrame:
        if self.textsDataFrame is not None:
            return self.textsDataFrame.loc[self.examplesLoc[_cls]]

    def getTree(self, tree: Tree | None = None, parent: str | None = None) -> Tree:
        if tree is None:
            tree = Tree()
        tree.create_node(
            f'{self.name} [{" | ".join(map(str, [len(self.examplesLoc[_cls]) for _cls in [1, 2, 0]]))}]', self.name, parent=parent)
        for child in self.children.values():
            child.getTree(tree, parent=self.name)
        return tree

    def __repr__(self) -> str:
        repr = f'TaxonomyNode\nName: {self.name}\nParent: {self.parent.name if self.parent is not None else None}\nChildren: {list(self.children.keys())}\nExamples Loc: {self.examplesLoc}\n'
        return repr


def getTaxonomy(
        taxonomyDataFrame: pd.DataFrame,
        textsDataFrame: pd.DataFrame | None = None,
) -> tuple[TaxonomyNode, Tree]:
    taxonomy = TaxonomyNode('Таксономия', textsDataFrame=textsDataFrame)
    for _, row in taxonomyDataFrame.iterrows():
        curNode = taxonomy
        for task in row.dropna():
            if task not in curNode.children:
                newNode = TaxonomyNode(
                    name=task, parent=curNode, textsDataFrame=textsDataFrame)
                curNode.addChild(newNode)
            else:
                newNode = curNode.getChild(task)
            curNode = newNode
    _ = taxonomy.setExamplesLoc()
    tree = taxonomy.getTree()
    return taxonomy, tree


if __name__ == '__main__':
    taxonomyDataFrame = pd.read_excel('data/raw/Тестовое задание.xlsx')
    textsDataFrame = pd.read_excel(
        'data/raw/Тестовое задание.xlsx', sheet_name='Тестовые данные')

    taxonomy, taxonomyTree = getTaxonomy(
        taxonomyDataFrame=taxonomyDataFrame,
        textsDataFrame=textsDataFrame,
    )
    print(taxonomyTree.show(stdout=False))
