import json
import os

from scipy import spatial


def computeEmbeddings(
    model,
    texts: list[str],
    keys: list[str],
    jsonPath: str,
    recompute: bool = False
):
    assert len(texts) == len(keys)

    if os.path.exists(jsonPath) and not recompute:
        with open(jsonPath, 'r') as jsonfile:
            data = jsonfile.read()
        embeddingsDict = json.loads(data)
        assert set(keys) == set(embeddingsDict.keys())
        return embeddingsDict

    texts = [text.replace("\n", " ") for text in texts]
    embeddings = model.embed_documents(texts=texts)
    embeddingsDict = {key: embedding for key,
                      embedding in zip(keys, embeddings)}
    with open(jsonPath, "w", encoding='utf-8') as jsonfile:
        json.dump(embeddingsDict, jsonfile, ensure_ascii=False)

    return embeddingsDict


def classifyText(
    text: str,
    # model,
    tasksEmbeddings: dict[str, list[float]],
    textsEmbeddings: dict[str, list[float]] | None = None,
    top_n: int = 1,
):
    if (textsEmbeddings is not None) and (text in textsEmbeddings):
        textEmbedding = textsEmbeddings[text]
    else:
        pass
        # text = text.replace("\n", " ")
        # textEmbedding = model.embed_documents(texts=[text])[0]

    tasksSimilarities = dict()
    for task, taskEmbedding in tasksEmbeddings.items():
        similarity = 1 - \
            spatial.distance.cosine(textEmbedding, taskEmbedding)
        tasksSimilarities[task] = similarity

    return sorted(tasksSimilarities, key=tasksSimilarities.get)[-top_n:][::-1]
