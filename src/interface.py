import torch
import json
import os
from modelo import ModeradorCNN

def carregar_interface():
    print("========================================")
    print(" 🛡️ RBooster - Filtro de Moderação 🛡️")
    print("========================================")
    
    if not os.path.exists("vocabulario.json") or not os.path.exists("pesos/pesos_moderador.pth"):
        print("Erro: Os arquivos de treino não foram encontrados!")
        return

    # 1. Carrega o Dicionário (Os "óculos" da IA)
    with open("vocabulario.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    print(f"✅ Dicionário carregado ({len(vocab)} caracteres)")
    
    # 2. Reconstrói a Arquitetura CNN (Agora com 4 janelas!)
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=32, num_filtros=64)
    
    # 3. Injeta a Memória do Treinamento
    modelo.load_state_dict(torch.load("pesos/pesos_moderador.pth", weights_only=True))
    modelo.eval() # Coloca a IA em Modo de Leitura (Garante que ela não altere os pesos)
    print("✅ IA Armada e Operacional!\n")
    
    print("Digite os testes (máximo 100 caracteres).")
    print("Experimente xingamentos camuflados ou frases normais.")
    print("Digite 'sair' para encerrar.\n")
    
    while True:
        texto = input("💬 Frase: ")
        
        if texto.lower() == 'sair':
            break
            
        if len(texto) == 0:
            continue
            
        # 4. Tradução Visual (Letras -> IDs)
        texto_str = texto.lower()
        sequencia = []
        
        for char in texto_str:
            sequencia.append(vocab.get(char, vocab["<UNK>"]))
            
        # 5. Aplica a Régua Certa (Padding/Truncamento para 100 caracteres)
        max_len = 100
        if len(sequencia) > max_len:
            sequencia = sequencia[:max_len]
        elif len(sequencia) < max_len:
            sequencia = sequencia + [vocab["<PAD>"]] * (max_len - len(sequencia))
            
        # Transforma no tensor para a IA (Adiciona a dimensão do Lote)
        tensor_entrada = torch.tensor([sequencia], dtype=torch.long)
        
        # 6. Pede o Julgamento
        with torch.no_grad(): # Desativa as matemáticas de treinamento por velocidade
            predicao = modelo(tensor_entrada)
            probabilidade = predicao.item() * 100
            
        # 7. Regras de Negócio do seu Servidor (Exemplo)
        if probabilidade >= 75.0:
            status = "🔴 PROIBIDO / BANIMENTO"
        elif probabilidade >= 40.0:
            status = "🟡 SUSPEITO / REVISÃO HUMANA"
        else:
            status = "🟢 SEGURO"
            
        print("-" * 45)
        print(f"☢️ Índice de Toxicidade: {probabilidade:.2f}%")
        print(f"⚖️ Decisão do Sistema: {status}")
        print("-" * 45 + "\n")

if __name__ == "__main__":
    carregar_interface()
