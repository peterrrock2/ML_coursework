import unittest
import sys
import numpy as np


class KernelTester(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        self.X1 = np.random.normal(0, 1, (2, 2))
        self.X2 = np.random.normal(0, 1, (2, 2))

        self.comment = f"Testing kernel {'{}'}\n X1, X2:{self.X1, self.X2}"
        self.linear = np.array([[-0.08393451, 0.67830853],
                                [-0.50825603, 2.19166405]])
        self.polynomial = np.array([[0.76874015, 4.7273244],
                                    [0.11890966, 32.51258597]])
        self.radial = np.array([[0.33732183, 0.01860754],
                                [0.0004392, 0.05628524]])

    def test_kernel(self, target, obtained):
        # with self.subTest(i=1):
        self.assertTrue(np.allclose(obtained, getattr(self, target), atol=1e-3), msg=
        f"{self.comment.format(target)}\n obtained:{obtained}\n expected: {getattr(self, target)}")

    def test_linear(self):
        self.test_kernel("linear", LinearKernel().compute(self.X1, self.X2))

    def test_radial(self):
        self.test_kernel("radial", RadialKernel(2).compute(self.X1, self.X2))

    def test_poly(self):
        self.test_kernel("polynomial", PolynomialKernel(1, 3).compute(self.X1, self.X2))


class LinearTester(unittest.TestCase):

    def setUp(self) -> None:
        self.features = np.array([[1.55143777, 0.2644804, 0.0995576],
                                  [0.22541014, 1.6967911, -0.45701382],
                                  [0.12528546, -1.44263567, 0.7017054],
                                  [-1.30567135, -0.86010032, -1.13522536]])
        self.labels = np.array([136.70039877, 10.1003086, 44.67363091, -221.48398972])
        self.comment = f"Testing {'{}'} {'{}'}, alpha =2, normalize={'{}'} \n X:  {str(self.features)} \ntargets: {str(self.labels)}"

        self.true_values = {"ridge": {"coefficients": {False: np.array([67.3816571, 12.4267024, 46.63028522]),
                                                       True: np.array([59.85861897, 18.28561265, 48.08714515])},
                                      "intercept": {False: -7.2683820675025785, True: -7.502412860000002}
                                      },
                            "lasso": {"coefficients": {False: np.array([83.36110924, 15.21050409, 79.08888918]),
                                                       True: np.array([82.32525113, 19.25895655, 56.15183344])},
                                      "intercept": {False: -2.9950281444221063, True: -7.502412860000005}
                                      },
                            "elastic": {"coefficients": {False: np.array([53.27862827, 10.94209836, 33.96771899]),
                                                         True: np.array([48.11601119, 14.15394196, 38.74380055])},
                                        "intercept": {False: -7.796083758223746, True: -7.502412860000006}
                                        }
                            }

    def get_coef(self, cls, name, normalize, target):
        model__ = cls(alpha=2, normalize=normalize)

        expected = self.true_values[name][target][normalize]
        model__.fit(self.features, self.labels)
        obtained = getattr(model__, target)
        comment = self.comment.format(name, target, normalize)
        comment = f"{comment} \n expected: {expected} \n obtained: {obtained}"

        self.assertTrue(np.allclose(expected, obtained, atol=1e-6), msg=comment)

    def test_ridge_coef1(self):
        self.get_coef(Ridge, "ridge", True, "coefficients")

    def test_ridge_coef2(self):
        self.get_coef(Ridge, "ridge", False, "coefficients")

    def test_ridge_intercept1(self):
        self.get_coef(Ridge, "ridge", True, "intercept")

    def test_ridge_intercept2(self):
        self.get_coef(Ridge, "ridge", False, "intercept")

    def test_lasso_coef1(self):
        self.get_coef(Lasso, "lasso", True, "coefficients")

    def test_lasso_coef2(self):
        self.get_coef(Lasso, "lasso", False, "coefficients")

    def test_lasso_intercept1(self):
        self.get_coef(Lasso, "lasso", True, "intercept")

    def test_lasso_intercept2(self):
        self.get_coef(Lasso, "lasso", False, "intercept")

    def test_elastic_coef1(self):
        self.get_coef(ElasticNet, "elastic", True, "coefficients")

    def test_elastic_coef2(self):
        self.get_coef(ElasticNet, "elastic", False, "coefficients")

    def test_elastic_intercept1(self):
        self.get_coef(ElasticNet, "elastic", True, "intercept")

    def test_elastic_intercept2(self):
        self.get_coef(ElasticNet, "elastic", False, "intercept")


class NaiveBayesTester(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array([[0, 1, 0, 1],
                                  [0, 1, 0, 0],
                                  [1, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 1, 1, 1]])
        self.labels = np.array([1, 0, 1, 0, 0])
        self.classes_log = np.array([-0.51082562, -0.91629073])
        self.features_log = {i: v for i, v in enumerate([np.array([[-0.40546511, -1.09861229],
                                                                   [-0.69314718, -0.69314718]]),
                                                         np.array([[-1.09861229, -0.40546511],
                                                                   [-0.69314718, -0.69314718]]),
                                                         np.array([[-0.40546511, -1.09861229],
                                                                   [-0.69314718, -0.69314718]]),
                                                         np.array([[-1.09861229, -0.40546511],
                                                                   [-0.69314718, -0.69314718]])])}
        self.jll = np.array([[-1.62186043, -2.77258872],
                             [-2.31500761, -2.77258872],
                             [-4.39444915, -2.77258872],
                             [-2.31500761, -2.77258872],
                             [-3.00815479, -2.77258872]])
        self.predict = np.array([0, 0, 1, 0, 0])

        self.comment = f"Testing {'{}'} {'{}'}\n X:  {str(self.features)} \ntargets: {str(self.labels)}"

    def test_classes(self):
        name = "naivebayes"
        target = "classes_log_probability"
        nb__ = NaiveBayes().fit(self.features, self.labels)
        expected = self.classes_log
        obtained = nb__.classes_log_probability

        comment = self.comment.format(name, target)
        comment = f"{comment} \n expected: {expected} \n obtained: {obtained}"

        self.assertTrue(np.allclose(expected, obtained, atol=1e-6), msg=comment)

    def test_features(self):
        name = "naivebayes"
        target = "features_log_likelihood"
        nb__ = NaiveBayes().fit(self.features, self.labels)
        expected = self.features_log
        obtained = nb__.features_log_likelihood

        comment = self.comment.format(name, target)
        comment = f"{comment} \n expected: {expected} \n obtained: {obtained}"

        self.assertTrue(all([np.allclose(expected[i], obtained[i], atol=1e-4) for i in expected]), msg=comment)

    def test_jll(self):
        name = "naivebayes"
        target = "joint_log_likelihood"
        nb__ = NaiveBayes().fit(self.features, self.labels)
        expected = self.jll
        obtained = nb__.joint_log_likelihood(self.features)

        comment = self.comment.format(name, target)
        comment = f"{comment} \n expected: {expected} \n obtained: {obtained}"

        self.assertTrue(np.allclose(expected, obtained, atol=1e-6), msg=comment)

    def test_predict(self):
        name = "naivebayes"
        target = "predict"
        nb__ = NaiveBayes().fit(self.features, self.labels)
        expected = self.predict
        obtained = nb__.predict(self.features)

        comment = self.comment.format(name, target)
        comment = f"{comment} \n expected: {expected} \n obtained: {obtained}"

        self.assertTrue(np.allclose(expected, obtained, atol=1e-6), msg=comment)


test = sys.argv[1]

all_tests = {
    "kernels": (KernelTester, ["test_linear", "test_radial", "test_poly"]),
    "ridge": (LinearTester, ["test_ridge_coef1", "test_ridge_coef2", "test_ridge_intercept1", "test_ridge_intercept2"]),
    "lasso": (LinearTester, ["test_lasso_coef1", "test_lasso_coef2", "test_lasso_intercept1", "test_lasso_intercept2"]),
    "elastic": (
        LinearTester,
        ["test_elastic_coef1", "test_elastic_coef2", "test_elastic_intercept1", "test_elastic_intercept2"]),
    "naivebayes": (NaiveBayesTester, ["test_classes", "test_features", "test_jll", "test_predict"])
}
suite = unittest.TestSuite()

C, tests = all_tests[test]
for t in tests:
    suite.addTest(C(t))
runner = unittest.TextTestRunner(verbosity=1).run(suite)
