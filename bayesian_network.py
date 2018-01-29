import pandas as pd
import numpy as np
import pomegranate as pg
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import (ParameterEstimator,
                              MaximumLikelihoodEstimator,
                              BayesianEstimator,
                              BdeuScore,
                              K2Score,
                              BicScore,
                              ExhaustiveSearch,
                              HillClimbSearch,
                              ConstraintBasedEstimator)

DATASET_PATH = '/Users/g.sarapulov/MLProjects/lnt/data/dataset.csv'
DEMOGRAPHIC_VARIABLES = [1, 2]
BASKET_VARIABLES = [324, 326]  # [3, 326]
RECOMMENDATIONS_VARIABLES = [649, 649]  # [327, 650]


def get_dataset():
    # data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))
    # data['A'] += data['B'] + data['C']
    # data['H'] = data['G'] - data['A']
    # data['E'] *= data['F']
    data = pd.read_csv(DATASET_PATH)
    return data


def split_variables(data):
    i, j = DEMOGRAPHIC_VARIABLES
    demographic_variables = data.columns[i:j+1]
    i, j = BASKET_VARIABLES
    basket = data.columns[i:j+1]
    i, j = RECOMMENDATIONS_VARIABLES
    recommendations = data.columns[i:j+1]
    dependencies = []
    # for x in demographic_variables:
    #     for y in basket:
    #         dependencies.append((x, y))
    for y in basket:
        for z in recommendations:
            dependencies.append((y, z))

    return dependencies


def get_model(dependencies):
    # model = BayesianModel([('A', 'C'), ('A', 'B'), ('C', 'B'), ('G', 'A'), ('G', 'H'), ('H', 'A')])
    model = BayesianModel(dependencies)
    return model


def estimate_parameters(model, data):
    pe = ParameterEstimator(model, data)
    return pe


def estimate_mle(model, data):
    # contribute smoothing
    mle = MaximumLikelihoodEstimator(model, data)
    return mle


def estimate_bayes(model, data):
    est = BayesianEstimator(model, data)
    # print(est.estimate_cpd('tasty', prior_type='BDeu', equivalent_sample_size=1))
    return est


def fit_model(data, model):
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=1)
    # for cpd in model.get_cpds():
    #     print(cpd)
    return model


def model_score_bdeu(model, data):
    return BdeuScore(data).score(model)


def model_score_bic(model, data):
    return BicScore(data).score(model)


def exhaustive_search(model, data):
    bic = model_score_bic(model, data)
    es = ExhaustiveSearch(data, scoring_method=bic)
    best_model = es.estimate()
    print(best_model.edges())

    print("\nAll DAGs by score:")
    for score, dag in reversed(es.all_scores()):
        print(score, dag.edges())


def constraints_based_search(model, data):

    est = ConstraintBasedEstimator(data)

    print(est.test_conditional_independence('B', 'H'))  # dependent
    print(est.test_conditional_independence('B', 'E'))  # independent
    print(est.test_conditional_independence('B', 'H', ['A']))  # independent
    print(est.test_conditional_independence('A', 'G'))  # independent
    print(est.test_conditional_independence('A', 'G', ['H']))  # dependent

if __name__ == '__main__':
    data = get_dataset()
    vars = split_variables(data)
    print(vars)
    model = get_model(vars)
    # print(BayesianEstimator(model, data).estimate_cpd('Yoghurts_', prior_type='K2', equivalent_sample_size=0))
    print(model.get_cpds())
    infer = VariableElimination(model)
    # print(infer.query(['Yoghurts_'], evidence={'age': 'age1', 'Yoghurts': 1})['Yoghurts_'])
    print(infer.map_query(['Yoghurts_']))
    model = pg.BayesianNetwork
    # print(MaximumLikelihoodEstimator(model, data).estimate_cpd('Yoghurts_'))
    # model.fit(data, estimator=BayesianEstimator)
    # for cpd in model.get_cpds()[:3]:
    #     print(cpd)
    # constraints_based_search(model, data)
    # print(model_score_bic(model, data))
