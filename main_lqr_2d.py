import gpytorch
import torch
from src.TVBO import TimeVaryingBOModel
from src.objective_functions_LQR import lqr_objective_function_2D


# parameters regarding the objective function
objective_function_options = {'objective_function': lqr_objective_function_2D,

                              # optimize the last two entries of the feedback gain
                              'spatio_dimensions': 2,

                              # approximate noise level from the objective function
                              'noise_lvl': 0.005,

                              # feasible set for the optimization
                              'feasible_set': torch.tensor([[-62.5, -5], [-12.5, -1]],
                                                           dtype=torch.float),

                              # initial feasible set consisting of only controllers
                              'initial_feasible_set': torch.tensor([[-50, -4], [-25, -2]],
                                                                   dtype=torch.float),

                              # scaling \theta to have approximately equal lengthscales in each dimension
                              'scaling_factors': torch.tensor([3, 1 / 4])}

# parameters regarding the model
model_options = {
                 # 'constrained_dimensions': None,  # later specified for each variation
                 # 'forgetting_type': None,  # later specified for each variation
                 # 'forgetting_factor': None,  # later specified for each variation

                 # specification for the constraints  (cf. Agrell 2019)
                 'nr_samples': 10000,
                 'xv_points_per_dim': 4,  # VOPs per dimensions
                 'truncation_bounds': [0, 2],

                 # specification of prior
                 'prior_mean': 0.,
                 'lengthscale_constraint': gpytorch.constraints.Interval(0.5, 6),
                 'lengthscale_hyperprior': gpytorch.priors.GammaPrior(6, 1 / 0.3),
                 'outputscale_constraint_spatio': gpytorch.constraints.Interval(0, 20),
                 'outputscale_hyperprior_spatio': None, }



# UI -> UI-TVBO, B2P_OU -> TV-GP-UCB
variations = [
    # in the paper color green
    {'forgetting_type': 'UI', 'forgetting_factor': 0.03, 'constrained_dims': [0, 1]},

    # in the paper color blue
    {'forgetting_type': 'UI', 'forgetting_factor': 0.03, 'constrained_dims': []},

    # in the paper color purple
    {'forgetting_type': 'B2P_OU', 'forgetting_factor': 0.03, 'constrained_dims': [0, 1]},

    # in the paper color red
    {'forgetting_type': 'B2P_OU', 'forgetting_factor': 0.03, 'constrained_dims': []}, ]


trials_per_variation = 25  # number of different runs
for variation in variations:

    # update variation specific parameters
    # model_options['forgetting_type'] = variation['forgetting_type']
    # model_options['forgetting_factor'] = variation['forgetting_factor']
    # model_options['constrained_dimensions'] = variation['constrained_dims']

    tvbo_model = TimeVaryingBOModel(objective_function_options=objective_function_options,
                                    model_options=model_options,
                                    post_processing_options={},
                                    add_noise=False, )  # noise is added during the simulation of the pendulum

    # specify name to safe results
    method_name = model_options['forgetting_type']
    forgetting_factor = model_options['forgetting_factor']
    string = 'constrained' if model_options['constrained_dimensions'] else 'unconstrained'
    NAME = f"{method_name}_2DLQR_{string}_forgetting_factor_{forgetting_factor}".replace('.', '_')

    # run optimization
    for trial in range(1, trials_per_variation + 1):
        tvbo_model.run_TVBO(n_initial_points=30,
                            time_horizon=300,
                            safe_name=NAME,
                            trial=trial, )
    print('Finished.')