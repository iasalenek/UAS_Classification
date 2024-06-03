import pandas as pd
from .embeddings import classifyText
from .taxonomy import TaxonomyNode


def topNTasksAccuracy(
    textsDataFrame: pd.DataFrame,
    tasksEmbeddings: dict[str, list[float]],
    textsEmbeddings: dict[str, list[float]],
    top_n: int = 1,
):
    incorrectIdx = []
    correct = 0
    for idx, (text, label, _cls) in textsDataFrame.iterrows():
        if _cls == 1:
            tasks = classifyText(
                text=text,
                tasksEmbeddings=tasksEmbeddings,
                textsEmbeddings=textsEmbeddings,
                top_n=top_n,
            )
            if label in tasks:
                correct += 1
            else:
                incorrectIdx.append(idx)
    return correct / sum(textsDataFrame['Cls'] == 1), incorrectIdx


def partiallyCorrectAcuracy(
        taxonomy: TaxonomyNode,
        textsDataFrame: pd.DataFrame,
        tasksEmbeddings: dict[str, list[float]],
        textsEmbeddings: dict[str, list[float]],
):
    incorrectIdx = []
    leafs = taxonomy.getLeafs()
    correct = 0
    for idx, (text, label, _cls) in textsDataFrame.iterrows():
        if _cls != 0:
            task = classifyText(
                text=text,
                tasksEmbeddings=tasksEmbeddings,
                textsEmbeddings=textsEmbeddings,
            )[0]
            if label in leafs[task].parent.children.keys():
                correct += 1
            else:
                incorrectIdx.append(idx)
    return correct / sum(textsDataFrame['Cls'] != 0), incorrectIdx
