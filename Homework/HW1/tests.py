import unittest
import sys
import numpy as np


class TestKNNC(unittest.TestCase):

    def setUp(self):
        self.features = np.array([[1, 1], [1, 2], [2, 1], [5, 2], [3, 2], [8, 2], [2, 4]])
        self.labels = np.array([1, -1, -1, 1, -1, 1, -1])
        self.test_points = np.array([[1, 1.1], [3, 1], [7, 5], [2, 6], [4, 4]])
        self.ins = "\n X:" + str(self.features) + ",\nlabels:" + str(self.labels)
        self.topic = "Testing KNN(3) "
        self.knn_3 = KNNClassifier(3).fit(self.features, self.labels)

    def test_majority(self):
        # print("majority vote test")
        majority = np.array([-1, 1])
        comment = self.topic + "majority_vote" + self.ins + "\n expected output: " + str(majority)
        obtained = self.knn_3.majority_vote(np.array([[1, 2, 3], [3, 4, 5]]), np.array([[.1, .2, .3], [.1, .2, .3]]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(majority, obtained), msg=comment)

    def test_predict(self):
        test_labels = np.array([-1, -1, 1, -1, -1])
        comment = f"{self.topic}  predict {self.ins} \n expected output: {test_labels}"
        obtained = self.knn_3.predict(self.test_points)
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(test_labels, obtained), msg=comment)

    def test_confusion(self):
        confusion = np.array([[2., 1.], [2., 0.]])
        comment = f"{self.topic} confusion {self.ins} \n expected output:  {confusion}"
        obtained = self.knn_3.confusion_matrix(self.test_points, np.array([1, -1, -1, 1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(confusion, obtained), msg=comment)

    def test_accuracy(self):
        # print("Accuracy test")
        accuracy = 0.6
        comment = self.topic + "accuracy" + self.ins + "\n expected output: " + str(accuracy)
        obtained = self.knn_3.accuracy(self.test_points, np.array([1, 1, 1, -1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, accuracy), msg=comment)


class TestWKNNC(unittest.TestCase):

    def setUp(self) -> None:
        self.features = np.array([[1, 1], [1, 2], [2, 1], [5, 2], [3, 2], [8, 2], [2, 4]])
        self.labels = np.array([1, -1, -1, 1, -1, 1, -1])
        self.test_points = np.array([[1, 1.1], [3, 1], [7, 5], [2, 6], [4, 4]])
        self.test_labels = np.array([1, -1, 1, -1, -1])
        self.ins = "\n X:" + str(self.features) + ",\nlabels:" + str(self.labels)
        self.topic = "Testing WeightedKNN(3) "
        self.knn_3 = WeightedKNNClassifier(3).fit(self.features, self.labels)

    def test_vote(self):
        majority = np.array([-1, 1])
        comment = self.topic + "weighted_vote" + self.ins + "\n expected output: " + str(majority)
        obtained = self.knn_3.weighted_vote(np.array([[1, 2], [3, 4]]), np.array([[0.1, 0.2], [1, 2]]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, majority, atol=1e-5), msg=comment)

    def test_predict(self):
        comment = self.topic + "predict" + self.ins + "\n expected output: " + str(self.test_labels)
        obtained = self.knn_3.predict(self.test_points)
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, self.test_labels, atol=1e-5), msg=comment)


test = sys.argv[1]

all_tests = {"knnc": (TestKNNC, ["test_majority", "test_predict", "test_confusion", "test_accuracy"]),
             "weightedknnc": (TestWKNNC, ["test_vote", "test_predict"])
             }
suite = unittest.TestSuite()

C, tests = all_tests[test]
for t in tests:
    suite.addTest(C(t))
runner = unittest.TextTestRunner(verbosity=1).run(suite)
