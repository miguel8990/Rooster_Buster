import torch
import json
import os
import sys
import time
from modelo import ModeradorCNN

def rodar_benchmark(modelo, vocab):
    print("\n🚀 Iniciando Benchmark de Precisão e Velocidade (50 Frases)...")
    
    frases = [
        # 25 Frases Boas e Limpas (Esperado < 0.50)
        # Inclui frases armadilha (viés) com nomes de minorias em contexto neutro
        ("O dia hoje está maravilhoso", 0),
        ("Parabéns pelo excelente trabalho no projeto", 0),
        ("Eu amo comer pizza no fim de semana", 0),
        ("Gays merecem respeito igual a todos", 0), 
        ("Negros, brancos, índios, somos todos iguais", 0), 
        ("Essa receita de bolo ficou ótima", 0),
        ("Muito obrigado pela ajuda", 0),
        ("Sua didática é incrível professor", 0),
        ("O sol está lindo hoje", 0),
        ("Gostei bastante desse livro", 0),
        ("Meu cachorro aprendeu a dar a pata", 0),
        ("Vamos ao cinema mais tarde?", 0),
        ("Bom dia grupo, que a semana seja abençoada", 0),
        ("Feliz aniversário, muita paz e saúde", 0),
        ("Tamo junto meu irmão, sempre que precisar", 0),
        ("A reunião foi muito produtiva", 0),
        ("Adorei a nova atualização do jogo", 0),
        ("Que paisagem belíssima", 0),
        ("Essa música me traz boas memórias", 0),
        ("Comprei um presente pra minha mãe", 0),
        ("A paz mundial é o que todos queremos", 0),
        ("Fiquei muito feliz com a notícia", 0),
        ("Excelente atendimento ao cliente", 0),
        ("O transporte público funcionou bem hoje", 0),
        ("Minha florzinha abriu no jardim", 0),
        
        # 25 Frases Tóxicas e Pesadas (Esperado >= 0.50)
        # Variações de xingamentos, ódio, racismo e xenofobia
        ("Vai se foder seu lixo humano", 1),
        ("Sua mãe é uma vadia", 1),
        ("Vtnc some da minha frente", 1),
        ("Se mata logo, inútil", 1),
        ("Preto tem que ir pra senzala", 1), 
        ("Viado escroto morre logo", 1), 
        ("Mulher só serve pra lavar louça", 1), 
        ("Seu fdp do caralho", 1),
        ("Você é a escória da sociedade", 1),
        ("Vou te quebrar na porrada seu merda", 1),
        ("Lix0 inútil que não sabe fazer nada", 1),
        ("Gordo nojento vai fazer dieta", 1),
        ("Te estupro se te ver na rua", 1), 
        ("Nordestino vem roubar emprego nosso", 1), 
        ("Arrombado vai chupar um canavial de rola", 1),
        ("Cala a boca seu doente mental", 1),
        ("Pobre vagabundo fica pedindo", 1),
        ("Quero que você exploda, otário", 1),
        ("Seu trabalho é uma bosta absoluta", 1),
        ("Vai dar o rabo viadinho", 1),
        ("Vagabunda de esquina", 1),
        ("Covarde covarde covarde lixo", 1),
        ("Corno fodido", 1),
        ("Sua família toda devia morrer", 1),
        ("Imbecil, não tem cérebro não?", 1)
    ]
    
    t0 = time.time()
    acertos, falsos_positivos, falsos_negativos = 0, 0, 0
    
    for txt, label_real in frases:
        txt_str = txt.lower()
        sequencia = [vocab.get(char, vocab["<UNK>"]) for char in txt_str]
        
        if len(sequencia) > 300:
            sequencia = sequencia[:300]
        else:
            sequencia = sequencia + [vocab["<PAD>"]] * (300 - len(sequencia))
            
        tensor = torch.tensor([sequencia], dtype=torch.long)
        with torch.no_grad():
            predicao_bruta = modelo(tensor)
            pred = torch.sigmoid(predicao_bruta).item()
            
        label_pred = 1 if pred >= 0.5 else 0
        if label_pred == label_real:
            acertos += 1
        elif label_pred == 1 and label_real == 0:
            falsos_positivos += 1
            print(f"⚠️ [FALSO POSITIVO] A IA baniu um inocente: '{txt}' ({pred*100:.1f}%)")
        else:
            falsos_negativos += 1
            print(f"⚠️ [FALSO NEGATIVO] A IA deixou passar a ofensa: '{txt}' ({pred*100:.1f}%)")
            
    t1 = time.time()
    
    print("\n📊 RESULTADOS DO BENCHMARK (50 Frases) 📊")
    print(f"⏱️  Tempo total: {(t1 - t0)*1000:.2f} ms ({(t1 - t0)*1000/50:.2f} ms por frase na CPU)")
    print(f"🎯 Acurácia Geral: {(acertos/50)*100:.1f}% ({acertos}/50 certos)")
    print(f"❌ Falsos Positivos (Inocentes punidos): {falsos_positivos}/25")
    print(f"❌ Falsos Negativos (Ódio não punido):   {falsos_negativos}/25")
    print("=" * 45 + "\n")

def carregar_interface():
    print("========================================")
    print(" 🛡️ RBooster - Filtro de Moderação 🛡️")
    print("========================================")
    
    if not os.path.exists("vocabulario.json") or not os.path.exists("pesos/pesos_moderador.pth"):
        print("Erro: Os arquivos de treino (vocabulario.json e pesos_moderador.pth) não foram encontrados!")
        print("Rode 'python3 src/treinador.py' primeiro para gerar a rede neural.")
        return

    with open("vocabulario.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    print(f"✅ Dicionário carregado ({len(vocab)} caracteres)")
    
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=64, num_filtros=128)
    
    # O torch.compile (magia negra) adiciona '_orig_mod.' no nome das camadas ao salvar.
    # Precisamos limpar esse prefixo para a rede original aceitar os pesos.
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True)
    state_dict_limpo = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict_limpo)
    
    modelo.eval()
    print("✅ Cérebro Artificial Armado e Operacional!\n")
    
    escolha = input("Deseja rodar o Benchmark Rápido primeiro? (s/n): ").strip().lower()
    if escolha == 's':
        rodar_benchmark(modelo, vocab)
    
    print("--- MODO INTERATIVO ---")
    print("Digite os testes (máximo 300 caracteres).")
    print("Digite 'sair' para encerrar.\n")
    
    while True:
        texto = input("💬 Frase: ")
        
        if texto.lower() == 'sair':
            break
            
        if len(texto) == 0:
            continue
            
        texto_str = texto.lower()
        sequencia = [vocab.get(char, vocab["<UNK>"]) for char in texto_str]
        
        max_len = 300
        if len(sequencia) > max_len:
            sequencia = sequencia[:max_len]
        elif len(sequencia) < max_len:
            sequencia = sequencia + [vocab["<PAD>"]] * (max_len - len(sequencia))
            
        tensor_entrada = torch.tensor([sequencia], dtype=torch.long)
        
        with torch.no_grad():
            predicao_bruta = modelo(tensor_entrada)
            probabilidade = torch.sigmoid(predicao_bruta).item() * 100
            
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
