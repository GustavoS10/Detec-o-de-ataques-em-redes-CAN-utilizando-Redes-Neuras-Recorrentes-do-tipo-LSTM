# Detecção de Ataques em Redes CAN Utilizando LSTM

Este repositório contém o código, dados e instruções para o artigo **"Detecção de Ataques em Redes CAN: Um Estudo de Caso Aplicado a Veículos Autônomos"**. Este estudo visa aprimorar a segurança de redes CAN (Controller Area Network) em veículos autônomos, utilizando Redes Neurais Recorrentes (RNN) do tipo Long Short-Term Memory (LSTM) para detecção de anomalias e ataques cibernéticos.

## 📄 Resumo do Artigo

As redes CAN são amplamente usadas para comunicação em veículos, especialmente autônomos, porém são suscetíveis a diversos tipos de ataques cibernéticos. Este estudo propõe uma abordagem baseada em LSTM para detectar padrões temporais que indiquem anomalias e ataques, como DoS, Fuzzy, e Impersonation. O modelo foi treinado e testado em um conjunto de dados real de tráfego CAN, alcançando uma precisão de 97% e provando ser uma solução eficaz para a detecção de anomalias em redes CAN.

## 📊 Principais Resultados

- **Acurácia**: 97%
- **F1-Score**: 0.95 (alta precisão e Recall)
- **Capacidade de Generalização**: Capaz de detectar desvios de tráfego anômalo com base em padrões históricos, minimizando falsos positivos e falsos negativos.
- **Hardware Utilizado para Treinamento**: NVIDIA GeForce RTX 3090 com TensorFlow 2.9.0.

## 🔧 Pré-requisitos

- **Python 3.8 ou superior**
- **Bibliotecas**: TensorFlow, NumPy, pandas, scikit-learn, matplotlib
