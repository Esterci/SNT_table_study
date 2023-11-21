from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
import numpy as np
import time
import pandas as pd


def sliding_window_MLP_Regressor(
    dataset, layers, batch, function, iterr, learningRate, beta1, beta2, epocas, shuffle
):
    window_size = int(0.3 * len(dataset))
    window_inf = 0
    window_sup = window_size
    window = 1

    global_error = []
    start_time = time.time()  # Início da contagem de tempo

    while window_sup <= len(dataset):
        tf.random.set_seed(0)
        # tf.config.optimizer.set_jit(True)
        lista_window_erros = {}
        dados = dataset[window_inf:window_sup]

        # Definindo a ordem do sistema
        X = pd.concat(
            [dados.shift(1), dados.shift(2), dados.shift(3), dados.shift(4)], axis=1
        )
        y = pd.concat([dados.shift(-4)], axis=1)

        X.dropna(subset=["Daily Power yields (kWh)"], inplace=True)
        y.dropna(subset=["Daily Power yields (kWh)"], inplace=True)

        X = X.to_numpy()
        y = y.to_numpy()

        y = y.flatten()

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, shuffle=False, stratify=None
        # )
        # X_test, X_val, y_test, y_val = train_test_split(
        #     X_temp, y_temp, test_size=0.5, shuffle=False, stratify=None
        # )
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, shuffle=False, stratify=None
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False, stratify=None
        )

        # Aplicando o modelo MLP Regressor

        # Conferindo o lag:
        input_sequence_length = len(X[0])
        # print(input_sequence_length)
        input_layer = tf.keras.layers.Input(shape=(input_sequence_length,))

        mlp = input_layer  # Inicializar o modelo com a camada de entrada

        for neurons in layers:
            mlp = tf.keras.layers.Dense(units=neurons, activation=function)(mlp)
            mlp = tf.keras.layers.Dropout(0.2)(mlp)

        # Camada de saída com output_dim=1
        # output_layer = tf.keras.layers.Dense(1, activation="linear")(mlp)
        output_layer = tf.keras.layers.Dense(1, activation="linear")(mlp)

        # Definir o modelo
        mlp_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=iterr,
            verbose=1,
            min_delta=0.0001,
            mode="min",
            restore_best_weights=True,
        )
        opt = tf.keras.optimizers.Adam(
            learning_rate=learningRate, beta_1=beta1, beta_2=beta2
        )

        # Compilar o modelo
        mlp_model.compile(loss="mean_squared_error", optimizer=opt)

        mlp_model.fit(
            x=X_train,
            y=y_train,
            batch_size=int(batch),
            epochs=epocas,
            verbose=1,
            callbacks=[early_stopping_callback],
            # validation_split=0.0,
            validation_data=(X_val, y_val),
            shuffle=shuffle,
            workers=-1,
            use_multiprocessing=True,
        )

        OutputTest = mlp_model.predict(X_test, workers=-1, use_multiprocessing=True)

        local_error = [
            ((previsto - target) ** 2) ** 0.5
            for previsto, target in zip(OutputTest, y_test)
        ]

        global_error += local_error
        window_inf = window_inf + len(OutputTest)
        window_sup = window_sup + len(OutputTest)
        window += 1
        K.clear_session()

    rmse = np.mean(global_error)
    std = np.std(global_error)

    end_time = time.time()  # Fim da contagem de tempo

    save_time = end_time - start_time  # Tempo total para salvar
    hp_str = "layers_{}__function_{}__batch_{}__iter_{}__LearnR_{}__Beta1_{}__Beta2_{}__time_{}".format(
        layers, function, batch, iterr, learningRate, beta1, beta2, save_time
    )

    np.savetxt(
        "resultadosAnt/" + hp_str + ".csv",
        np.array([rmse, std]),
        delimiter=",",
    )
