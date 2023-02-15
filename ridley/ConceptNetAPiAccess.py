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
    first_id = ""
    for i in edges:
        id = i["end"]["@id"]
        first_id = id
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
    maxN = neighbors[first_id]
    for n in neighbors.keys():
        if neighbors[n] > maxVal:
            maxVal = neighbors[n]
            maxN = n

    return maxN

def GetAllCommonNeighbors(entity):
    first = GetEdges(entity)
    edges = first["edges"]
    neighbors = {}
    for i in edges:
        id = i["end"]["@id"]
        id2 = i["start"]["@id"]
        split = id.split("/")
        split2 = id2.split("/")
        if id[:5] != '/c/en' or id == first["@id"]:
            continue
        else:
            split = id.split("/")
            if len(split) < 7:
                ids = [split[3]]
            else:
                ids = [split[3], split[6]]
            for u in ids:
                if u not in neighbors:
                    neighbors[u] = 1
                else:
                    neighbors[u] += 1

        if id2[:5] != '/c/en' or id2 == first["@id"]:
            continue
        else:
            split = id2.split("/")
            if len(split) < 7:
                ids = [split[3]]
            else:
                ids = [split[3], split[6]]
            for u in ids:
                if u not in neighbors:
                    neighbors[u] = 1
                else:
                    neighbors[u] += 1
    keys = neighbors.copy()
    for k in keys.keys():
        commons = GetEdges(k)["edges"]
        for c in commons:
            id = c["end"]["@id"]
            id2 = c["start"]["@id"]
            if id[:5] != '/c/en' or id == first["@id"]:
                break
            else:
                split = id.split("/")
                if len(split) < 7:
                    ids = [split[3]]
                else:
                    ids = [split[3], split[6]]
                for u in ids:
                    if u not in neighbors:
                        neighbors[u] = 1
                    else:
                        neighbors[u] += 1

            if id2[:5] != '/c/en' or id2 == first["@id"]:
                break
            else:
                split = id2.split("/")
                if len(split) < 7:
                    ids = [split[3]]
                else:
                    ids = [split[3], split[6]]
                for u in ids:
                    if u not in neighbors:
                        neighbors[u] = 1
                    else:
                        neighbors[u] += 1

    total = []
    for k in neighbors.keys():
        total.append(k.replace("_", " "))
    return total