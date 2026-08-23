import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import json
import os
from modelo import ModeradorCNN, contar_parametros

def preparar_dataset():
    print("Carregando bases filtradas...")
    df_told = pd.read_csv("dados/told_br_curto.csv")
    df_hate = pd.read_csv("dados/hatecheck_curto.csv")
    
    # O TOLD-BR já tem a coluna 'label' com 0 (seguro) e 1 (tóxico)
    # Não precisamos mexer.
    
    # HateCheck: 'hateful' vira 1.0, 'non-hateful' vira 0.0
    df_hate['label'] = df_hate['label_gold'].apply(lambda x: 1.0 if x == 'hateful' else 0.0)
    
    # Junta os dois universos de dados
    textos = df_told['text'].tolist() + df_hate['test_case'].tolist()
    rotulos = df_told['label'].tolist() + df_hate['label'].tolist()
    
    print(f"Total de frases prontas para treino: {len(textos)}")
    return textos, rotulos

def construir_vocabulario(textos):
    print("Mapeando todas as letras e símbolos do universo...")
    caracteres_unicos = set()
    for txt in textos:
        for char in str(txt):
            caracteres_unicos.add(char.lower())
    
    # ID 0 = PADDING (Espaço Vazio no final da frase)
    # ID 1 = UNK (Desconhecido - Letras alienígenas que surgirem no futuro)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for char in sorted(list(caracteres_unicos)):
        vocab[char] = idx
        idx += 1
        
    print(f"Vocabulário criado com apenas {len(vocab)} caracteres diferentes!")
    
    # Salvamos o dicionário, pois a Interface de Teste vai precisar dele para ler textos novos
    with open("vocabulario.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    return vocab

def textos_para_tensor(textos, rotulos, vocab, max_len=100):
    entradas = []
    gabaritos = []
    
    for txt, rotulo in zip(textos, rotulos):
        txt_str = str(txt).lower()
        sequencia = []
        
        # Converte cada letra pro seu ID matemático
        for char in txt_str:
            sequencia.append(vocab.get(char, vocab["<UNK>"]))
            
        # Garante que todos os arrays tenham cravados 100 de tamanho
        if len(sequencia) > max_len:
            sequencia = sequencia[:max_len]
        elif len(sequencia) < max_len:
            sequencia = sequencia + [vocab["<PAD>"]] * (max_len - len(sequencia))
            
        entradas.append(sequencia)
        gabaritos.append([rotulo])
        
    return torch.tensor(entradas, dtype=torch.long), torch.tensor(gabaritos, dtype=torch.float32)

def treinar_moderador():
    textos, rotulos = preparar_dataset()
    vocab = construir_vocabulario(textos)
    
    # Transforma texto humano em matemática de matriz
    x_dados, y_dados = textos_para_tensor(textos, rotulos, vocab, max_len=100)
    
    dataset = TensorDataset(x_dados, y_dados)
    # Batch de 256. Como a rede é microscópica, a CPU mastiga isso rápido.
    carregador = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Inicializa o Cérebro
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=32, num_filtros=64)
    contar_parametros(modelo)
    
    # Binary Cross Entropy: É a Loss perfeita para respostas de "Sim/Não" (0.0 ou 1.0)
    criterio = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    epocas = 10
    print("Iniciando a Caçada por Padrões Tóxicos!")
    
    for epoca in range(epocas):
        perda_acumulada = 0.0
        
        for lote_x, lote_y in carregador:
            predicao = modelo(lote_x)
            perda = criterio(predicao, lote_y)
            
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            
            perda_acumulada += perda.item()
            
        perda_media = perda_acumulada / len(carregador)
        print(f"Época [{epoca+1}/{epocas}] | Perda Média (BCELoss): {perda_media:.4f}")
        
    torch.save(modelo.state_dict(), "pesos_moderador.pth")
    print("\n✅ Rede treinada com sucesso! Pesos salvos em 'pesos_moderador.pth'")

if __name__ == "__main__":
    treinar_moderador()
