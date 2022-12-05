import unittest

from ridley.ridley.ConceptNetAPiAccess import (GetEntity, GetEdgeWithID, GetEdges, GetRelatedness, GetRelatedEnglish, GetEdgesBetween, GetCommonNeighbor)

class TestConceptNet(unittest.TestCase):
    def setUp(self):
        self.firstEntity = "dog"
        self.secondEntity = "cat"

    def test_get_entity(self):
        resp = GetEntity(self.firstEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(resp['@id'], "/c/en/dog")

    def test_get_edges_with_id(self):
        resp = GetEdgeWithID("/c/en/dog")
        self.assertIsNotNone(resp)
        self.assertEqual(len(resp), 5)

    def test_get_edges(self):
        resp = GetEdges(self.firstEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(len(resp), 5)

    def test_get_relatedness(self):
        resp = GetRelatedness(self.firstEntity, self.secondEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(resp, 0.558)

    def test_get_related_english(self):
        resp = GetRelatedEnglish(self.firstEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(len(resp), 50)

    def test_get_edges_between(self):
        resp = GetEdgesBetween(self.firstEntity, self.secondEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(len(resp), 4)

    def test_get_common_neighbor(self):
        resp = GetCommonNeighbor(self.firstEntity)
        self.assertIsNotNone(resp)
        self.assertEqual(resp, "/c/en/animal")

if __name__ == "__main__":
    unittest.main()