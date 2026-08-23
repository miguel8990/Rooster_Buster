import torch
import json
import os
import sqlite3
from modelo import ModeradorCNN

def rlhf_humano():
    print("========================================")
    print(" 🧠 RBooster - RLHF (Treino com Humano)")
    print("========================================")
    
    if not os.path.exists("vocabulario.json") or not os.path.exists("pesos/pesos_moderador.pth"):
        print("Erro: Os arquivos de treino não foram encontrados.")
        return

    with open("vocabulario.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=32, num_filtros=64)
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True)
    state_dict_limpo = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict_limpo)
    modelo.eval()
    
    caminho_csv = "dados/dados_sinteticos.csv"
    
    print("✅ Sistema Híbrido Ativado!")
    print("Digite a frase. Se a IA errar, corrija-a. O erro virará dado de treino.\n")
    
    while True:
        texto = input("💬 Frase de Teste (ou 'sair'): ").strip()
        
        if texto.lower() == 'sair':
            break
        if not texto:
            continue
            
        sequencia = [vocab.get(char, vocab["<UNK>"]) for char in texto.lower()]
        
        max_len = 300
        if len(sequencia) > max_len:
            sequencia = sequencia[:max_len]
        elif len(sequencia) < max_len:
            sequencia = sequencia + [vocab["<PAD>"]] * (max_len - len(sequencia))
            
        tensor = torch.tensor([sequencia], dtype=torch.long)
        
        with torch.no_grad():
            pred = torch.sigmoid(modelo(tensor)).item()
            
        probabilidade = pred * 100
        previsao_binaria = 1 if pred >= 0.5 else 0
        
        status = "🔴 TÓXICO" if previsao_binaria == 1 else "🟢 SEGURO"
        
        print(f"🤖 Previsão da IA: {status} ({probabilidade:.1f}%)")
        
        feedback = input("A IA acertou? [s/n]: ").strip().lower()
        
        if feedback == 'n':
            # Se a resposta é binária (0 ou 1) e ela errou, a reposta correta é obviamente o oposto matemático!
            real_val = 1 - previsao_binaria
            rotulo_nome = "TÓXICO (1)" if real_val == 1 else "SEGURO (0)"
            mensagem = f"🩸 ERRO CORRIGIDO AUTOMATICAMENTE PARA {rotulo_nome}! A IA aprenderá essa nova regra."
        else:
            # Se ela acertou, a previsão dela é o gabarito correto
            real_val = previsao_binaria
            mensagem = "✨ ACERTO CONFIRMADO! Dado guardado para reforçar a memória."
            
        # Grava no SQLite 100% das interações
        try:
            conn_db = sqlite3.connect('banco/dataset.db')
            cursor = conn_db.cursor()
            cursor.execute(
                "INSERT INTO frases (text, label, origem) VALUES (?, ?, ?)",
                (texto, real_val, 'sintetico_humano')
            )
            conn_db.commit()
            conn_db.close()
        except Exception as db_err:
            print(f"⚠️ Erro ao salvar no banco: {db_err}")
            
        print(f"💾 {mensagem}")
        print("-" * 45)

if __name__ == "__main__":
    rlhf_humano()
