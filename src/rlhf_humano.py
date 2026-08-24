import torch
import json
import os
import time
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
        
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=64, num_filtros=64)
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True, map_location='cpu')
    state_dict_limpo = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict_limpo)
    modelo.eval()
    
    print("\n📜 RLHF Humano (Reinforcement Learning from Human Feedback) - RBooster")
    print("Digite 'sair' a qualquer momento para voltar à vida real.")
    print("O modelo será avaliado. Se errar, nós corrigiremos e injetaremos a vacina no SQLite.\n")

    # Conexão SQLite persistente
    conn_db = sqlite3.connect('banco/dataset.db')

    try:
        while True:
            texto = input("💬 Digite uma frase para a CNN analisar: ").strip()
            
            if texto.lower() == 'sair':
                break
                
            if not texto:
                continue

            # Processamento
            sequencia = [vocab.get(char, vocab["<UNK>"]) for char in texto.lower()]
            if len(sequencia) > 300:
                sequencia = sequencia[:300]
            else:
                sequencia = sequencia + [vocab["<PAD>"]] * (300 - len(sequencia))
                
            tensor = torch.tensor([sequencia], dtype=torch.long)
            
            # Inferência
            t0 = time.time()
            with torch.no_grad():
                pred = torch.sigmoid(modelo(tensor)).item()
            latencia = (time.time() - t0) * 1000
            
            del tensor # Limpeza higiênica da memória
            
            classe = 1 if pred >= 0.5 else 0
            emoji = "🔴 TÓXICO" if classe == 1 else "🟢 SEGURO"
            
            print(f"🤖 Previsão da IA: {emoji} ({(pred*100):.1f}% de chance de toxicidade)")
            print(f"⏱️  Tempo de resposta: {latencia:.2f}ms")
            
            # Feedback
            acertou = input("A IA acertou? (s/n): ").strip().lower()
            
            rotulo_correto = classe
            if acertou == 'n':
                # Corrige invertendo matematicamente a classe errada
                rotulo_correto = 1 - classe
                print(f"💉 Aplicando vacina! Ensinando que essa frase é {'Tóxica' if rotulo_correto == 1 else 'Segura'}.")
            elif acertou != 's':
                print("⚠️ Comando não reconhecido. Assumindo que a IA acertou.")
                
            # Salva no Banco de Dados
            try:
                cursor = conn_db.cursor()
                cursor.execute(
                    "INSERT INTO frases (text, label, origem) VALUES (?, ?, ?)",
                    (texto, rotulo_correto, 'rlhf_humano')
                )
                conn_db.commit()
                print("✅ Dado salvo no banco de dados!")
            except Exception as e:
                print(f"❌ Erro ao salvar no banco: {e}")
                
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nSessão de treinamento RLHF finalizada pelo usuário.")
    finally:
        conn_db.close()

if __name__ == "__main__":
    rlhf_humano()
