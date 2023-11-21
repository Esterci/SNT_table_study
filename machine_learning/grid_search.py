import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from machine_learning.model_builder import sliding_window_MLP_Regressor
import pickle


hidden_layer_sizes = [
    (256, 256, 256, 256, 256),
    (300, 300, 300),
    (300, 300),
    (64, 64, 64, 64, 64),
    (64, 64, 64, 64),
    (64, 64, 64),
    (64, 64),
]
activation = [
    "softmax",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "softplus",
    "softsign",
    "elu",
]

batch_size = [200, 300, 400]
n_iter_no_change = [10]
learningRates = [0.01, 0.001, 0.0001, 0.00001]
beta_1_List = [0.98, 0.99]
beta_2_List = [0.98, 0.99]

state_file = "state.pkl"
try:
    with open(state_file, "rb") as f:
        state = pickle.load(f)
        last_completed_iteration = state.get("last_completed_iteration", 0)
except FileNotFoundError:
    state = {}
    last_completed_iteration = 0
for_iter = 0
for layers in hidden_layer_sizes:
    for function in activation:
        for batch in batch_size:
            for iterr in n_iter_no_change:
                for learningRate in learningRates:
                    for beta1 in beta_1_List:
                        for beta2 in beta_2_List:
                            # Pule iterações já concluídas
                            if last_completed_iteration > 0:
                                last_completed_iteration -= 1
                                continue
                            else:
                                # Atualize o estado após cada iteração
                                for_iter += 1
                                state["last_completed_iteration"] = for_iter
                                with open(state_file, "wb") as f:
                                    pickle.dump(state, f)
                                print("iteração: ", for_iter)
                                print(
                                    "Parameters: ",
                                    layers,
                                    function,
                                    batch,
                                    iterr,
                                    learningRate,
                                    beta1,
                                    beta2,
                                )
                                sliding_window_MLP_Regressor(
                                    Data_Power,
                                    layers,
                                    batch,
                                    function,
                                    iterr,
                                    learningRate,
                                    beta1,
                                    beta2,
                                    200,
                                    False,
                                )
