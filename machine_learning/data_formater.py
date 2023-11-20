import pandas as pd
import argparse

# defining hyper-parameters constructor

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--file",
    type=int,
    action="store",
    dest="file",
    required=True,
    help="File.",
)

args = parser.parse_args()

def data_format(file):
    df = pd.read_csv(file)
    df = df[df.delta_t > 0]
    df = df.drop(columns=["regiao", "data_da_inscricao", "data_do_evento"])

    df_bin = df.fillna(0)

    bin_cols = [
        "uf",
        "tipo_de_doador",
        "uf_origem",
        "sexo",
        "grupo_sanguineo",
        "cor",
    ]

    for col in bin_cols:
        # converting to binary data
        df_one = pd.get_dummies(
            df[col],
        )

        # Use the .replace() method to map True/False to 1/0

        df_one[df_one == True] = 1
        df_one[df_one == False] = 0

        # Recolocando NaN
        df_one.loc[df[col].isnull()] = None

        # Renomeando colunas
        new_columns = [
            col + "_" + category.strip().replace(" ", "_")
            for category in df_one.columns
        ]

        df_one.columns = new_columns

        df_aux = pd.concat([df_bin, df_one], axis=1)

        df_aux = df_aux.drop(columns=[col])

        df_bin = df_aux

    df_prev = df_bin[df_bin.transplante_bin == 1]
    df_prev = df_prev.drop(columns=["obito_bin", "transplante_bin"])

    df_prev = df_prev.dropna(
        subset=["tipo_de_doador_Doador_Falecido", "tipo_de_doador_Doador_Vivo"],
        how="any",
    )

    df_prev.to_csv("df_prev.csv")

    return 0

data_format(args["file"])