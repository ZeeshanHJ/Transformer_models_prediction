
"""
==================
2. ML Experiments
==================
"""


import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from utils import make_data
from ai4water.experiments import MLRegressionExperiments

data, _, _, _, _ = make_data()

print(data.shape)

#%%%%
# Initialize the experiment
comparisons = MLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    seed=1575,
    verbosity=1,
    show=True,
    save=True
)

#%%%%

# fit/train all the ML models

comparisons.fit(
    data=data,
    run_type="optimize",
    num_iterations=100,
    exclude=['model_LassoLarsIC', 'NuSVR', 'OneClassSVM',
             'SGDRegressor', 'SVR', 'TheilsenRegressor'],
    #          'AdaBoostRegressor', 'LinearSVR',
    #          'BaggingRegressor', 'DecisionTreeRegressor',
    #          'HistGradientBoostingRegressor',
    #          'ExtraTreesRegressor', 'ExtraTreeRegressor',
    #          'LinearRegression', 'KNeighborsRegressor', 'RandomForestRegressor']
)


#%%%%
# Compare R2

_ = comparisons.compare_errors(
    'r2',
    data=data)
plt.tight_layout()
plt.show()

#%%%
# Compare MSE

_ = comparisons.compare_errors(
    'mse',
    data=data,
    cutoff_val=1e7,
    cutoff_type="less"
)
plt.tight_layout()
plt.show()
#
# # Best models with higher R2 values
#
_ = best_models = comparisons.compare_errors(
    'r2_score',
    cutoff_type='greater',
    cutoff_val=0.01,
    data=data
)
plt.tight_layout()
plt.show()

# # Tayler plot comperison
#
comparisons.taylor_plot(data=data)
plt.show()
#
# #%%%%%
# #EDF plot
comparisons.compare_edf_plots(
    data=data,
    exclude=["OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV",
             "ElasticNet", "ElasticNetCV", "PoissonRegressor",
             "RadiusNeighborsRegressor", "Lars", "RANSACRegressor",
             "HuberRegressor", "LinearSVR", "LGBMRegressor",
             "LinearRegression", "MLPRegressor", "RandomForestRegressor",
             "RidgeCV", "Ridge", "TweedieRegressor"]
    )

plt.tight_layout()
plt.show()
#
#
# #%%%%
# # Regression Plot
_ = comparisons.compare_regression_plots(data=data, figsize=(12, 14))
plt.show()
#
# #%%%%
# # Residual plot
_ = comparisons.compare_residual_plots(data=data, figsize=(12, 14))
plt.show()
