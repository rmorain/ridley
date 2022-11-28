import requests

base_uri = "http://api.conceptnet.io"


def GetEntity(entity):
    words = entity.split(" ")
    uri = "/uri?language=en&text="
    ending = ""
    for word in words:
        ending += word + "+"
    ending = ending[: (len(ending) - 1)]
    full_uri = base_uri + uri + ending
    return requests.get(full_uri).json()


def GetEdgeWithID(entity):
    full_edge_uri = base_uri + entity
    return requests.get(full_edge_uri).json()


def GetEdges(entity):
    response = GetEntity(entity)
    full_edge_uri = base_uri + response["@id"]

    return requests.get(full_edge_uri).json()


def GetRelatedness(entity1, entity2):
    id1 = GetEntity(entity1)["@id"]
    id2 = GetEntity(entity2)["@id"]
    full_uri = base_uri + "/relatedness?node1=" + id1 + "&node2=" + id2
    return requests.get(full_uri).json()["value"]


def GetRelatedEnglish(entity):
    id = GetEntity(entity)["@id"]

    full_uri = base_uri + "/related" + id + "?filter=/c/en"
    return requests.get(full_uri).json()["related"]


def GetEdgesBetween(entity1, entity2):
    id1 = GetEntity(entity1)["@id"]
    id2 = GetEntity(entity2)["@id"]

    full_uri = base_uri + "/query?node=" + id1 + "&other=" + id2
    return requests.get(full_uri).json()


def GetCommonNeighbor(entity):
    edges = GetEdges(entity)["edges"]
    neighbors = {}
    for i in edges:
        id = i["end"]["@id"]
        if id not in neighbors:
            neighbors[id] = 1
        else:
            neighbors[id] += 1
    keys = neighbors.copy()
    for k in keys.keys():
        commons = GetEdgeWithID(k)["edges"]
        for c in commons:
            id = c["end"]["@id"]
            if id not in neighbors:
                neighbors[id] = 1
            else:
                neighbors[id] += 1

    maxVal = 0
    maxN = neighbors[id]
    for n in neighbors.keys():
        if neighbors[n] > maxVal:
            maxVal = neighbors[n]
            maxN = n

    return maxN


resp = GetCommonNeighbor("dog")

print(resp)
