import array
import pandas as pd
from deap import algorithms, base, creator, tools
from encoding_utils import *
from label_obj import labelObj
from probability_obj_boundary import boundaryProbabilityObj
from probability_obj_nonboundary import nonboundaryProbabilityObj
from distance_obj import distanceObj
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

class MOCE():
    def __init__(self,
                 dataset,
                 predict_fn=None,
                 predict_proba_fn=None,
                 boundary=False,
                 n_cf=5,
                 K_nbrs=500,
                 n_population=100,
                 n_generation=10,
                 crossover_perc=0.8,
                 mutation_perc=0.5,
                 hof_size=100,
                 init_x_perc=0.3,
                 init_neighbor_perc=0.6,
                 init_random_perc=1.0,
                 division_factor = 10,
                 ):

        self.dataset = dataset
        self.feature_names =  dataset['feature_names']
        self.n_features = len(dataset['feature_names'])
        self.feature_width = dataset['feature_width']
        self.continuous_indices = dataset['continuous_indices']
        self.discrete_indices = dataset['discrete_indices']
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.boundary = boundary
        self.n_cf = n_cf
        self.K_nbrs = K_nbrs
        self.n_population = n_population
        self.n_generation = n_generation
        self.crossover_perc = crossover_perc
        self.mutation_perc = mutation_perc
        self.hof_size = hof_size
        self.init_probability = [init_x_perc, init_neighbor_perc, init_random_perc] / \
                                np.sum([init_x_perc, init_neighbor_perc, init_random_perc])
        self.division_factor = division_factor
        self.objectiveFunction = self.constructObjectiveFunction()

    def constructObjectiveFunction(self):

        # print('Constructing objective function according to -boundary- hyper-parameter ...')

        if self.boundary == False:
            # objective names
            self.objective_names = ['label', 'probability', 'distance']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 3

            # defining objective function
            def objectiveFunction(x_ord, cf_class, probability_thresh, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_ord = theta2ord(cf_theta, featureScaler, dataset)

                # objective 1: label
                label_cost = labelObj(cf_ord, predict_fn, cf_class)

                # objective 2: probability
                probability_cost = nonboundaryProbabilityObj(cf_ord, predict_proba_fn, probability_thresh, cf_class)

                # objective 3: distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                return label_cost, probability_cost, distance_cost

            return objectiveFunction

        else:
            # objective names
            self.objective_names = ['label', 'probability', 'distance']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0 , -1.0)
            # number of objectives
            self.n_objectives = 3

            # defining objective function
            def objectiveFunction(x_ord, cf_class, probability_thresh, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_ord = theta2ord(cf_theta, featureScaler, dataset)

                # objective 1: label
                label_cost = labelObj(cf_ord, predict_fn, cf_class)

                # objective 2: probability
                probability_cost = boundaryProbabilityObj(cf_ord, predict_proba_fn, probability_thresh, cf_class)

                # objective 3: distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                return label_cost, probability_cost, distance_cost

            return objectiveFunction


    def groundtruthData(self):

        # print('Identifying correctly predicted training data for each class ...')

        groundtruth_data = {}

        pred_train = self.predict_fn(self.X_train)
        groundtruth_ind = np.where(pred_train == self.Y_train)
        pred_groundtruth = self.predict_fn(self.X_train[groundtruth_ind])

        self.n_classes = np.unique(self.Y_train)
        for c in self.n_classes:
            c_ind = np.where(pred_groundtruth == c)
            c_ind = groundtruth_ind[0][c_ind[0]]
            groundtruth_data[c] = self.X_train[c_ind].copy()

        return groundtruth_data

    def featureScaler(self):

        # print('Creating a scaler for mapping features to equal range ...')

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(self.X_train)

        return feature_scaler

    def neighborhoodModel(self):

        # print('Creating neighborhood models for every class of correctly predicted training data ...')

        neighborhood_models = {}
        for key, data in self.groundtruthData.items():
            data_ohe = ord2ohe(data, self.dataset)
            K_nbrs = min(self.K_nbrs, len(data_ohe))
            neighborhood_model = NearestNeighbors(n_neighbors=K_nbrs, algorithm='ball_tree', metric='minkowski', p=2)
            neighborhood_model.fit(data_ohe)
            neighborhood_models[key] = neighborhood_model
        return neighborhood_models

    def fit(self, X_train, Y_train):

        # print('Fitting the framework on the training data ...')

        self.X_train = X_train
        self.Y_train = Y_train

        self.groundtruthData = self.groundtruthData()
        self.featureScaler = self.featureScaler()
        self.neighborhoodModel = self.neighborhoodModel()

    # creating toolbox for optimization algorithm
    def setupToolbox(self, x_ord, x_theta, cf_class, probability_thresh, neighbor_theta):

        # print('Creating toolbox for the optimization algorithm ...')

        # initialization function
        def initialization(x_theta, neighbor_theta, n_features, init_probability):
            method = np.random.choice(['x', 'neighbor', 'random'], size=1, p=init_probability)
            if method == 'x':
                return list(x_theta)
            elif method == 'neighbor':
                idx = np.random.choice(range(len(neighbor_theta)), size=1)
                return list(neighbor_theta[idx].ravel())
            elif method == 'random':
                return list(np.random.uniform(0, 1, n_features))

        # creating toolbox
        toolbox = base.Toolbox()
        creator.create("FitnessMulti", base.Fitness, weights=self.objective_weights)
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
        toolbox.register("evaluate", self.objectiveFunction, x_ord, cf_class, probability_thresh, self.dataset,
                         self.predict_fn, self.predict_proba_fn, self.feature_width, self.continuous_indices,
                         self.discrete_indices, self.featureScaler)
        toolbox.register("attr_float", initialization, x_theta, neighbor_theta, self.n_features, self.init_probability)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / self.n_features)
        # toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        ref_points = tools.uniform_reference_points(len(self.objective_weights), self.division_factor)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        return toolbox

    # executing the optimization algorithm
    def runEA(self):

        # print('Running NSGA-III optimization algorithm ...')

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        hof = tools.HallOfFame(self.hof_size)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "avg", "max"

        pop = self.toolbox.population(n=self.n_population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(pop=pop, gen=0, evals=len(invalid_ind), **record)
        # print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self.n_generation):
            offspring = algorithms.varAnd(pop, self.toolbox, self.crossover_perc, self.mutation_perc)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, self.n_population)

            # Compile statistics about the new population
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(pop=pop, gen=gen, evals=len(invalid_ind), **record)
            # print(logbook.stream)

        fronts = tools.emo.sortLogNondominated(pop, self.n_population)

        return fronts, pop, hof, record, logbook

    # explain instance using multi-objective counterfactuals
    def explain(self,
                x_ord,
                cf_class='opposite',
                probability_thresh=0.5,
                ):

        # print('Generating counterfactual explanations ...')

        # finding the label of counterfactual instance
        if cf_class is 'opposite':
            x_class = self.predict_fn(x_ord.reshape(1,-1))
            cf_target = 1 - x_class[0]
            cf_class = cf_target
        elif cf_class is 'neighbor':
            x_proba = self.predict_proba_fn(x_ord.reshape(1,-1))
            cf_target = np.argsort(x_proba)[0][-2]
            cf_class = cf_target
        elif cf_class is 'strange':
            x_proba = self.predict_proba_fn(x_ord.reshape(1,-1))
            cf_target = np.argsort(x_proba)[0][0]
            cf_class = cf_target
        else:
            cf_target = cf_class

        # finding the neighborhood data of the counterfactual instance
        distances, indices = self.neighborhoodModel[cf_target].kneighbors(x_ord.reshape(1, -1))
        neighbor_data = self.groundtruthData[cf_target][indices[0]].copy()
        neighbor_theta = self.featureScaler.transform(neighbor_data)

        # creating toolbox for the counterfactual instance
        x_theta = ord2theta(x_ord, self.featureScaler)
        self.toolbox = self.setupToolbox(x_ord, x_theta, cf_class, probability_thresh, neighbor_theta)

        # running optimization algorithm for finding counterfactual instances
        fronts, pop, hof, record, logbook = self.runEA()

        # constructing counterfactuals
        cfs_theta = np.asarray([i for i in hof.items])
        cfs_ord = theta2ord(cfs_theta,self.featureScaler, self.dataset)
        cfs_ord = pd.DataFrame(data=cfs_ord, columns=self.feature_names)
        cfs_ord.drop_duplicates(keep="first", inplace=True)
        cfs_ord = cfs_ord.iloc[:self.n_cf, :]

        # selecting the best counterfactual
        best_cf_ord = cfs_ord.iloc[0]

        # evaluating counterfactuals
        cfs_theta = ord2theta(cfs_ord.to_numpy(), self.featureScaler)
        cfs_eval = np.asarray([np.asarray(self.toolbox.evaluate(cf)) for cf in cfs_theta])
        cfs_eval = pd.DataFrame(data=cfs_eval, columns=self.objective_names)

        # evaluation of the best counterfactual
        best_cf_eval = cfs_eval.iloc[0]
        # print(best_cf_eval, '\n')

        ## returning the results
        explanations = {'cfs_ord': cfs_ord,
                        'best_cf_ord': best_cf_ord,
                        'cfs_eval': cfs_eval,
                        'best_cf_eval': best_cf_eval
                        }

        return explanations