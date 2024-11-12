import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 #Adicionado recente
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE

# Redireciona todas as saídas do console para um arquivo de log
log_file = '/home/gsovrani/tccgs/logs.txt'
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout  # Opcional: redireciona erros para o mesmo arquivo

def is_hex(value):
    """Verifica se um valor é hexadecimal."""
    try:
        int(value, 16)
        return True
    except (ValueError, TypeError):
        return False

def hex_to_decimal(value):
    """Converte um valor hexadecimal para decimal."""
    try:
        return int(value, 16)
    except (ValueError, TypeError):
        return value  # Retorna o valor original se não for possível converter

def replace_missing_with_zero(df, columns):
    """Substitui valores ausentes por zero em colunas especificadas."""
    for col in columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace(['', ' '], 0)

def convert_to_numeric(df, columns):
    """Converte colunas especificadas para tipo numérico, coercivo para NaN onde falha."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def normalize_columns(df, columns):
    """Normaliza colunas especificadas entre 0 e 1."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return scaler

def load_and_process_data(file_path, columns_to_check, columns_to_normalize, target_column):
    # Carrega os dados
    df = pd.read_csv(file_path, dtype={col: str for col in columns_to_check}, low_memory=False)

    # Substitui valores ausentes por zero
    replace_missing_with_zero(df, columns_to_check)

    # Converte valores hexadecimais para decimais
    for col in columns_to_check:
        df[col] = df[col].apply(lambda x: hex_to_decimal(x) if is_hex(x) else x)
    
    # Converte as colunas para numéricas, forçando a conversão
    df = convert_to_numeric(df, columns_to_normalize)
    
    # Remove quaisquer linhas com valores NaN após conversão
    df.dropna(subset=columns_to_normalize, inplace=True)
    
    # Normaliza as colunas após garantir que todos os dados são numéricos
    scaler = normalize_columns(df, columns_to_normalize)
    
    return df, scaler

def create_sequences(data, target, time_steps=1):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)


def extract_features(model, X_data):
    # Certifique-se de que X_data tenha a forma adequada
    if len(X_data.shape) == 2:
        # Adiciona uma dimensão extra para features se necessário
        X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], 1))
    return model.predict(X_data)


def apply_tsne(X, n_components=3):
    """Aplica t-SNE nos dados."""
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(X)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

def save_model(model, model_path):
    """Salva o modelo no caminho especificado."""
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

def load_model(model_path):
    """Carrega o modelo salvo."""
    if os.path.exists(model_path):
        print(f"Carregando modelo salvo de: {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print("Modelo não encontrado. Treinando um novo...")
        return None

# def train_and_evaluate(X_seq, y_seq, test_size, time_steps, model_path=None):
    unique_classes = np.unique(y_seq)

    X_train, X_test, y_train, y_test = [], [], [], []

    # Divisão de dados por classe
    for cls in unique_classes:
        cls_indices = np.where(y_seq == cls)[0]
        cls_X = X_seq[cls_indices]
        cls_y = y_seq[cls_indices]

        cls_X_train, cls_X_test, cls_y_train, cls_y_test = train_test_split(
            cls_X, cls_y, test_size=test_size, shuffle=False
        )

        X_train.extend(cls_X_train)
        X_test.extend(cls_X_test)
        y_train.extend(cls_y_train)
        y_test.extend(cls_y_test)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Os dados de treinamento estão vazios. Verifique o processo de divisão dos dados.")

    # Transformação das saídas para o formato categórico
    y_train = tf.keras.utils.to_categorical(y_train)

    num_classes = y_train.shape[1]
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Carrega ou cria um novo modelo
    if model_path and os.path.exists(model_path):
        print(f"Carregando modelo salvo de: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Modelo não encontrado. Treinando um novo modelo...")
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000165362),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=1e-6)

        # Treinamento utilizando o mesmo conjunto de treinamento para validação
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=128,
            validation_data=(X_train, y_train),  # 28/10/24 alterei aqui de X_train e
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )

        # Curva de Convergência
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Perda de Treinamento', marker='o', linestyle='-', color='b')
        plt.plot(history.history['val_loss'], label='Perda de Validação', marker='o', linestyle='-', color='r')
        # plt.yscale('log')
        plt.title('Curva de Convergência - Escala Logarítmica')
        plt.xlabel('Épocas')
        plt.ylabel('Perda (Escala Logarítmica)')
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.savefig('/home/gsovrani/tccgs/graficoCC.png')
        plt.show()

        if model_path:
            model.save(model_path)
            print(f"Modelo salvo em: {model_path}")

    # Avaliação no próprio conjunto de treinamento
    loss, accuracy = model.evaluate(X_train, y_train)
    print(f'Treinamento - Loss: {loss}, Accuracy: {accuracy}')

    y_pred = model.predict(X_train)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)

    return y_train_classes, y_pred_classes, X_train

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

def train_and_evaluate(X_seq, y_seq, test_size, time_steps, model_path=None):
    unique_classes = np.unique(y_seq)

    X_train, X_test, y_train, y_test = [], [], [], []

    # Divisão de dados por classe
    for cls in unique_classes:
        cls_indices = np.where(y_seq == cls)[0]
        cls_X = X_seq[cls_indices]
        cls_y = y_seq[cls_indices]

        cls_X_train, cls_X_test, cls_y_train, cls_y_test = train_test_split(
            cls_X, cls_y, test_size=test_size, shuffle=False
        )

        X_train.extend(cls_X_train)
        X_test.extend(cls_X_test)
        y_train.extend(cls_y_train)
        y_test.extend(cls_y_test)

    # Converte listas para arrays numpy
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Verificação de cardinalidade
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Cardinalidade incompatível: X_train tem {X_train.shape[0]} amostras e y_train tem {y_train.shape[0]}. "
                         f"X_test tem {X_test.shape[0]} amostras e y_test tem {y_test.shape[0]}.")

    # Transformação das saídas para o formato categórico
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    num_classes = y_train.shape[1]
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Carrega ou cria um novo modelo
    if model_path and os.path.exists(model_path):
        print(f"Carregando modelo salvo de: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Modelo não encontrado. Treinando um novo modelo...")
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000165362),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)

        # Treinamento utilizando o conjunto de testes para validação
        history = model.fit(
            X_train, y_train,
            epochs=40,
            batch_size=128,
            validation_data=(X_test, y_test),  # Validação com o conjunto de testes
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )

        # Curva de Convergência
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Perda de Treinamento', marker='o', linestyle='-', color='b')
        plt.plot(history.history['val_loss'], label='Perda de Validação', marker='o', linestyle='-', color='r')
        plt.title('Curva de Convergência - Escala Logarítmica')
        plt.xlabel('Épocas')
        plt.ylabel('Perda (Escala Logarítmica)')
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.savefig('/home/gsovrani/tccgs/graficoCC.png')
        plt.show()

        if model_path:
            model.save(model_path)
            print(f"Modelo salvo em: {model_path}")

    # Avaliação no próprio conjunto de treinamento
    loss, accuracy = model.evaluate(X_train, y_train)
    print(f'Treinamento - Loss: {loss}, Accuracy: {accuracy}')



    y_pred = model.predict(X_train)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    # Cálculo do coeficiente de silhueta
    silhouette_avg = calculate_silhouette(X_test, y_test_classes)
    print(f"Silhueta {silhouette_avg}")
    # Chamada para gerar os gráficos t-SNE
    generate_tsne_plots(X_test, y_pred_classes, y_test)

    return y_train_classes, y_pred_classes, X_train


def calculate_silhouette(X_data, y_pred):
    """
    Calcula o coeficiente de silhueta para os dados preditos.
    :param X_data: Dados de entrada (pode ser as features originais ou reduzidas com t-SNE)
    :param y_pred: Classes preditas pelo modelo
    :return: Coeficiente de silhueta
    """
    # Garante que X_data está no formato correto (achatar se necessário)
    if len(X_data.shape) > 2:
        X_data_flat = X_data.reshape(X_data.shape[0], -1)
    else:
        X_data_flat = X_data

    # Calcula o coeficiente de silhueta
    silhouette_avg = silhouette_score(X_data_flat, y_pred)
    print(f"Coeficiente de Silhueta: {silhouette_avg}")
    return silhouette_avg

# def plot_tsne(X_tsne, labels, title, save_path):
#     """Gera um gráfico 2D do t-SNE."""
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10', s=60, alpha=0.7)
#     plt.title(title)
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Gráfico '{title}' salvo em: {save_path}")
def plot_tsne_3d(X_tsne, labels, title, colors):
    # Certifique-se de que 'colors' esteja no mesmo tamanho que X_tsne
    if len(labels) != X_tsne.shape[0]:
        labels = labels[:X_tsne.shape[0]]  # Ajusta o tamanho de labels para coincidir com X_tsne

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=labels if labels.size else 'b', cmap='viridis')
    plt.colorbar(scatter)
    ax.set_title(title)
    plt.show()


# def plot_tsne_3d(X_tsne, labels, title, name):
#     """Gera um gráfico t-SNE 3D com matplotlib."""
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Gera o gráfico de dispersão 3D
#     scatter = ax.scatter(
#         X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], 
#         c=labels, cmap='viridis', marker='o', s=50, alpha=0.7
#     )
    
#     # Adiciona uma barra de cores
#     legend = ax.legend(*scatter.legend_elements(), title="Classes")
#     ax.add_artist(legend)
    
#     # Títulos e labels
#     ax.set_title(title)
#     ax.set_xlabel("Componente 1")
#     ax.set_ylabel("Componente 2")
#     ax.set_zlabel("Componente 3")
#     plt.title(title)
#     plt.savefig(f"/home/gsovrani/tccgs/{name}")
#     # Exibe o gráfico
#     plt.show()

def generate_tsne_plots(X_test, y_pred_classes, y_test_classes):
    """Aplica t-SNE e gera gráficos 3D."""
    
    # Achata a entrada 3D em 2D: (amostras, features)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # (amostras, time_steps * features)

    # Aplica t-SNE nos dados achatados
    X_tsne = apply_tsne(X_test_flat, n_components=3)  # Reduz para 3 componentes

    print("Gerando primeiro gráfico")
    name1 = "tsne1"
    name2 = "tsne2"
    
    # Gráfico 1: Conjunto de Teste + y_pred (Predições)
    plot_tsne_3d(X_tsne, y_pred_classes, name1, 
                 "t-SNE: Conjunto de Teste + Predições")
    print("Gerando gráfico", name1)

    # Gráfico 2: Conjunto de Teste + y_true (Valores Verdadeiros)
    plot_tsne_3d(X_tsne, y_test_classes, name2, 
                 "t-SNE: Conjunto de Teste + Valores Verdadeiros")
    print("Gerando gráfico", name2)




# Caminho para salvar o modelo
model_path = "/home/gsovrani/tccgs/modelo_lstm.h5"

def confusion_matrix_gen(original, previstos, y):
    # Lista de rótulos descritivos
    class_labels = ['0 - Normal CAN message', '1 - DoS Attack', '2 - Fuzzy Attack', '3 - Impersonation Attack']
    # class_labels = ['0 - Normal CAN message', '1 - Impersonation', '2 - DoS']
    
    # Gera a matriz de confusão
    conf_matrix = confusion_matrix(original, previstos)
    
    # Exibe a matriz de confusão com rótulos descritivos
    plt.figure(figsize=(11, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusão Multiclass')
    plt.savefig('/home/gsovrani/tccgs/matriz_confusao.png')
    plt.show()

    # Gera o relatório de classificação com rótulos descritivos
    report = classification_report(original, previstos, target_names=class_labels)

    # Exibe o relatório de classificação
    print(report)
    
    # Salva o relatório de classificação em um arquivo de texto
    with open('/home/gsovrani/tccgs/classification_report.txt', 'w') as f:
        f.write(report)

def main(file_path, file_name, columns_to_check, columns_to_normalize, target_column, time_steps, test_sizes):
    df, scaler = load_and_process_data(file_path, columns_to_check, columns_to_normalize, target_column)

    # Salva o dataset convertido
    df.to_csv(file_name, index=False)

    # Carrega dados para processamento
    data = pd.read_csv(file_name)
    X = data[columns_to_normalize].values
    y = data[target_column].values

    # Cria sequências de dados para o modelo LSTM
    X_seq, y_seq = create_sequences(X, y, time_steps)
    # y_seq = np.where(y_seq == 3, 1, y_seq)  # Substitui o rótulo 3 por 1
    all_preds = []
    all_tests = []

    # Treinamento e avaliação para diferentes tamanhos de teste
    for test_size in test_sizes:
        print(f'\nTestando com {int((1 - test_size) * 100)}% treino e {int(test_size * 100)}% teste')
        y_test_classes, y_pred_classes, X_test = train_and_evaluate(X_seq, y_seq, test_size, time_steps, model_path)
        all_preds.extend(y_pred_classes)
        all_tests.extend(y_test_classes)

    class_labels = ['0 - Normal CAN message', '1 - DoS', '2 - Fuzzy', '3 - Impersonation']

    confusion_matrix_gen(np.array(all_tests), np.array(all_preds), class_labels)

# Parâmetros do processo
file_path = '/home/gsovrani/tccgs/dataset4.csv'
file_name = "/home/gsovrani/tccgs/dataset4-convertido.csv"
columns_to_check = ['TS','ID1', 'ID0', 'LEN', 'DLC0', 'DLC1', 'DLC2', 'DLC3', 'DLC4', 'DLC5', 'DLC6', 'DLC7']
columns_to_normalize = ['TS','ID1', 'ID0', 'LEN', 'DLC0', 'DLC1', 'DLC2', 'DLC3', 'DLC4', 'DLC5', 'DLC6', 'DLC7']
target_column = 'target'
time_steps = 2
test_sizes = [0.2]

# Verifica se TensorFlow está usando GPU
if tf.test.is_gpu_available():
    print("Treinando na GPU")
else:
    print("Treinando na CPU")

# Executa o processo
main(file_path, file_name, columns_to_check, columns_to_normalize, target_column, time_steps, test_sizes)