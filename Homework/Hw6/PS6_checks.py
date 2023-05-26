import numpy as np

class P1_Checker():
    def __init__(self):
        # Toy Datasets for Sanity Checks
        toy_X = np.array([[1,1], [-1,-1]])
        toy_y = [1,0]
        
        playground_X = np.array([[10,10],[0,0],[-1,-1]])
        playground_y = np.array([0,0,1])
    def P1_1_Check(self):
        #1.1 Sanity Check
        toy_model = KMeans(2)
        toy_model.initialize_centroids(toy_X)
        assert(toy_X[0] in toy_model.centroids)
        assert(toy_X[1] in toy_model.centroids)
    def P1_2_Check(self):
        #1.2 Sanity Check
        toy_model = KMeans(2)
        toy_model.initialize_centroids(toy_X)
        toy_distances = toy_model.compute_distances(toy_X)
        assert(np.array([2.82842712, 0.]) in toy_distances)
        assert(np.array([0., 2.82842712]) in toy_distances)
    def P1_3_Check(self):
        #1.3 Sanity Check
        #NOTE: Dependent on success with 1.2 - if distances are incorrect, nearest centroids may be incorrect
        #(This one isn't very thorough...)
        toy_model = KMeans(2)
        toy_model.initialize_centroids(toy_X)
        toy_distances = toy_model.compute_distances(toy_X)
        toy_assign = toy_model.compute_assignments(toy_distances)
        assert(toy_assign in np.array([[1,0],[0,1]]))
    def P1_4_Check(self):
        #1.4 Sanity Check
        toy_model = KMeans(2)
        toy_model.initialize_centroids(toy_X)
        #NOTE: Dependent on success with 1.2 and 1.3 - if assignments are incorrect, new centroids may be incorrect
        toy_distances = toy_model.compute_distances(toy_X)
        toy_assign = toy_model.compute_assignments(toy_distances)
        toy_new_cent = toy_model.compute_centroids(toy_X, toy_assign)
        assert(toy_X[0] in toy_new_cent)
        assert(toy_X[1] in toy_new_cent)
    def P1_5_Check(self):
        import math
        #1.5 Sanity Check
        toy_model = KMeans(2)
        toy_results = toy_model.fit(toy_X)
        assert(toy_results[0] == 0.0)

        playground_model = KMeans(2)
        playground_results = playground_model.fit(playground_X)
        assert(math.isclose(playground_results[-1], 1.0))
    def P1_6_Check(self):
        #1.6 Sanity Check
        #Requires 1.5 to work
        toy_model = KMeans(2)
        toy_model.fit(toy_X)
        toy_assign = toy_model.predict(np.array([[1.1,1],[-1.1,-1]]))
        assert(toy_assign in np.array([[1,0],[0,1]]))
class P2_Checker():
    def __init__(self):
        #Toy Dataset for Sanity Checks
        toy_reviews_X = ["David is the best professor !","David is bad ."]
        toy_reviews_y = [1,0]
    def P2_1_Check(self):
        #2.1 Sanity Check
        toy_transformer = TFIDF_Transformer()
        tokens = toy_transformer.tokenize(toy_reviews_X)
        tokens = toy_transformer.clear_stopwords(tokens)
        assert('is' not in tokens[0])
    def P2_2_Check(self):
        #2.2 Sanity Check
        toy_transformer = TFIDF_Transformer()
        tokens = toy_transformer.tokenize(toy_reviews_X)
        tokens = toy_transformer.clear_punct(tokens)
        assert('!' not in tokens[0])
    def P2_3_Check(self):
        #2.3 Sanity check
        toy_transformer = TFIDF_Transformer()
        tokens = toy_transformer.tokenize(toy_reviews_X)
        tokens = toy_transformer.clear_stopwords(tokens)
        tokens = toy_transformer.clear_punct(tokens)
        tokens = toy_transformer.stem(tokens)
        matrix = toy_transformer.count_matrix(tokens)
        assert(np.allclose(matrix,[[1.,1.,1.,0.],[1.,0.,0.,1.]]))
    def P2_4_Check(self):
        #2.4 Sanity check
        toy_transformer = TFIDF_Transformer()
        tokens = toy_transformer.tokenize(toy_reviews_X)
        tokens = toy_transformer.clear_stopwords(tokens)
        tokens = toy_transformer.clear_punct(tokens)
        tokens = toy_transformer.stem(tokens)
        matrix = toy_transformer.count_matrix(tokens)
        matrix = toy_transformer.tfidf_transform(matrix)
        assert(np.allclose(matrix,[[0,1,1,0],[0,0,0,1]]))