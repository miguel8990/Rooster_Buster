import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import sqlite3
import json
import os
import numpy as np
import random
import time
import logging
import resource

os.makedirs("Logs", exist_ok=True)

logging.basicConfig(
    filename=f"Logs/treinador{time.time():.0f}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Otimizações extremas para CPU Ryzen (Apenas Núcleos Físicos)
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.set_flush_denormal(True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from modelo import ModeradorCNN, contar_parametros

MAX_LEN = 300

# ─────────────────────────────────────────────────────────────────────────────
# O DATASET DE ALTA PERFORMANCE (SQLiteDataset)
# ─────────────────────────────────────────────────────────────────────────────
class SQLiteDataset(Dataset):
    """
    Esta classe traduz 4.3 milhões de textos para linguagem matemática
    ANTES do treino começar. Assim, o loop de treinamento não engasga.
    """
    def __init__(self, db_path, vocab, max_len=MAX_LEN):
        self.max_len = max_len
        
        unk_id = vocab.get("<UNK>", 1) # ID do caracter Desconhecido
        pad_id = vocab.get("<PAD>", 0) # ID de Preenchimento de espaços em branco

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # FASE 1: Contagem Leve
        # Ao invés de carregar o banco todo na RAM, usamos um comando SQL rápido
        # para descobrir apenas o TOTAL de frases válidas.
        total = cur.execute("SELECT COUNT(*) FROM frases WHERE CAST(label AS REAL) IS NOT NULL").fetchone()[0]
        print(f"📐 Total de linhas no banco: {total}")
        
        if len(vocab) > 32767:
            raise ValueError(f"Vocabulário ({len(vocab)}) excede o limite de int16 (32767). Risco de Overflow!")
            
        # FASE 2: A Mágica do NumPy (Evitando Memory Leak) + Pesos VIPs
        # O NumPy é escrito em C, logo ele dribla as frescuras de RAM do Python.
        # Nós criamos "Planilhas Gigantes Vazias" cheias de zeros.
        # int16 significa "número pequeno" (2 bytes). 
        # Isso faz 4.3M * 300 usar 2.6GB em vez de 10GB!
        self.X = np.zeros((total, max_len), dtype=np.int16)
        self.Y = np.zeros(total, dtype=np.float32)
        # NOVO: A Planilha de Privilégios (Sample Weighting)
        self.W = np.ones(total, dtype=np.float32) 
        
        ram_gb = (self.X.nbytes + self.Y.nbytes + self.W.nbytes) / (1024**3)
        print(f"📦 Arrays pré-alocados: ~{ram_gb:.2f} GB de RAM reservados.")
        
        # FASE 3: O Preenchimento (Streaming)
        print("🔄 Preenchendo arrays com dados do banco...")
        # O 'SELECT' sem 'fetchall' faz o SQLite cuspir os dados como uma mangueira.
        cur.execute("SELECT text, label, origem FROM frases WHERE CAST(label AS REAL) IS NOT NULL")
        
        idx = 0
        for (texto_raw, label_raw, origem) in cur:
            try:
                label = float(label_raw)
            except (TypeError, ValueError):
                continue
            
            texto = str(texto_raw).lower()
            
            # Nós pegamos a letra, achamos o número dela no dicionário
            # e carimbamos diretamente no quadrado vazio do Array NumPy!
            col = 0
            for c in texto:
                if col >= max_len:
                    break
                self.X[idx, col] = vocab.get(c, unk_id)
                col += 1
            
            self.Y[idx] = label
            
            # --- O SISTEMA DE PONTUAÇÃO E PRIVILÉGIOS (Sample Weighting) ---
            peso_amostra = 1.0 # Dados diluídos do povão valem peso 1x
            if origem:
                origem_str = str(origem).lower()
                if 'rlhf_humano' in origem_str:
                    peso_amostra = 8.0 # DADO OURO ABSOLUTO (As suas correções valem 10x mais)
                elif 'rlaif' in origem_str or 'sintetico' in origem_str:
                    peso_amostra = 2.0  # DADO PRATA (As batalhas da IA e as Vacinas valem 2x mais)
                    
            self.W[idx] = peso_amostra
            
            idx += 1
            if idx % 500_000 == 0:
                print(f"   ⏳ {idx}/{total} linhas processadas ({idx/total*100:.0f}%)...")
        
        # Caso alguma frase tenha falhado no Try/Except, cortamos a 'gordura' final do array vazio.
        if idx < total:
            self.X = self.X[:idx].copy()
            self.Y = self.Y[:idx].copy()
            self.W = self.W[:idx].copy()
        
        cur.close()
        conn.close()
        print(f"✅ Dataset pronto: {idx} amostras | RAM fixa: ~{ram_gb:.2f} GB")

    def __len__(self):
        """Diz ao PyTorch o tamanho total do nosso livro de dados."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Quando o PyTorch pede a frase número 'idx', nós recortamos aquela fatia.
        Agora entregamos 3 coisas: O Texto (X), o Gabarito (Y) e a Importância dela (W).
        Retornamos NumPy puro para evitar criar 3 objetos Tensor por amostra.
        O DataLoader converte tudo em Tensor de uma vez só no final (muito mais rápido).
        """
        return self.X[idx], np.array([self.Y[idx]]), np.array([self.W[idx]])


def construir_vocabulario(db_path):
    """
    Lê todo o banco de dados e cria um Dicionário de Letras Únicas (a, b, c, !, ?, emoji).
    O(1) Memory: Lê linha a linha pelo cursor sem armazenar nada além do dicionário final.
    """
    print("🔤 Construindo vocabulário (O(1) Memory)...")
    caracteres_unicos = set()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT text FROM frases")
    count = 0
    for (txt,) in cur:
        if txt:
            for char in str(txt).lower():
                caracteres_unicos.add(char)
        
        count += 1
        if count % 500_000 == 0:
            print(f"   ⏳ {count} textos escaneados para vocab...")

    cur.close()
    conn.close()
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, char in enumerate(sorted(caracteres_unicos), start=2):
        vocab[char] = i

    print(f"✅ Vocabulário: {len(vocab)} caracteres únicos.")
    with open("vocabulario.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab


def calcular_pos_weight(db_path):
    """
    O seu banco de dados tem 95% de frases Seguras e apenas 5% de Tóxicas.
    Se não ensinarmos a rede sobre isso, ela vai "chutar" Seguro sempre e acertar 95% da prova.
    Esse cálculo matemático cria um multiplicador para equilibrar o jogo.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            SUM(CASE WHEN CAST(label AS REAL) >= 0.5 THEN 1 ELSE 0 END),
            COUNT(*)
        FROM frases
        WHERE CAST(label AS REAL) IS NOT NULL
    """)
    toxicos, total = cur.fetchone()
    cur.close()
    conn.close()
    
    toxicos = toxicos or 0
    if total == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    limpos = total - toxicos
    
    # O cálculo original cru dava ~20x, o que deixava a rede paranóica.
    # Aplicando a raiz quadrada (** 0.5), nós derrubamos esse multiplicador para ~4.4x.
    # Isso diminui a punição por deixar passar um tóxico (False Negative) e
    # aumenta o peso relativo de "errar no geral" (False Positive em inocentes).
    proporcao_bruta = limpos / max(toxicos, 1)
    peso = proporcao_bruta ** 0.5 
    
    print(f"⚖️  pos_weight ajustado = {peso:.2f}x  (Proporção original era {proporcao_bruta:.2f}x)")
    return torch.tensor([peso], dtype=torch.float32)


def treinar_moderador():
    db_path = "banco/dataset.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Banco não encontrado: {db_path}. Rode a migração primeiro!")

    # O Mapeador de Hardware: Descobre se você tem uma Placa de Vídeo (GPU) ou usa o Processador (CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Treinando no hardware: {device.type.upper()}")

    # 1. Preparativos
    vocab = construir_vocabulario(db_path)
    peso_positivo = calcular_pos_weight(db_path).to(device)
    dataset = SQLiteDataset(db_path, vocab)

    # 2. O Entregador de Lotes (DataLoader)
    # Ele empacota as frases de 512 em 512.
    # num_workers=0: Como fatiamos numpy instantaneamente, abolimos os trabalhadores.
    
    # SPLIT DE VALIDAÇÃO (A "PROVA" DA IA)
    dataset_size = len(dataset)
    val_size = int(0.05 * dataset_size) # 5% escondidos
    train_size = dataset_size - val_size
    
    gerador = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=gerador)
    
    # Usando 2 ajudantes (workers) agora que o NumPy nos blindou contra o vazamento de memória (COW) do Linux
    carregador_treino = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, drop_last=True)
    carregador_val = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, drop_last=False)

    
    # 3. Criando o Cérebro
    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=64, num_filtros=128).to(device)
    contar_parametros(modelo)
    
    # 4. As Leis e Ferramentas de Treino
    # Criterio (Loss): O juiz. 'reduction=none' significa que o juiz não vai dar a nota média 
    # da sala sozinho. Ele vai entregar a nota de erro CADA frase solta nas nossas mãos.
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivo, reduction='none')
    
    # Otimizador (AdamW): O professor particular que mexe nos botões da rede.
    otimizador = optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.01)
    # Scheduler: Diminui o poder de mudança do professor aos poucos.
    scheduler = CosineAnnealingLR(otimizador, T_max=3, eta_min=1e-5)

    epocas = 3
    print("🥊 Iniciando a Caçada por Padrões Tóxicos!\n")
    os.makedirs("pesos", exist_ok=True)

    for epoca in range(epocas):
        modelo.train() # Coloca a rede em modo 'estudante ativo' (Habilita o Dropout de amnésia)
        perda_acumulada = 0.0
        acertos = 0
        total_amostras = 0

        t_dados_start = time.time()
        
        # O Loop Interno: Bate ponto e aprende! (Agora recebendo também o lote_w)
        for batch_idx, (lote_x, lote_y, lote_w) in enumerate(carregador_treino):
            t_dados = time.time() - t_dados_start
            t_comp_start = time.time()
            
            # Envia os dados para a CPU ou Placa de Vídeo (O que estiver livre)
            lote_x = lote_x.to(device, non_blocking=True).long() # Conversão de 1024 frases em bloco via C++ (Turbo)
            lote_y = lote_y.to(device, non_blocking=True)
            lote_w = lote_w.to(device, non_blocking=True)
            
            # Zera a memória de punições do professor do lote anterior
            otimizador.zero_grad(set_to_none=True)
            
            # Forward: O Estudante tenta adivinhar a resposta (Faz a prova)
            # Removemos a Precisão Mista (BFloat16), pois processadores sem AVX-512 emulam 
            # ela no software de forma letalmente lenta. Vamos usar Float32 Nativo.
            predicao_bruta = modelo(lote_x).view_as(lote_y)
            
            # Loss Bruta: O Juiz entrega a lista com os 1024 erros da prova separadamente.
            perda_bruta = criterio(predicao_bruta, lote_y)
            
            # Loss Ponderada: Multiplica o erro de cada frase pela sua "Importância" VIP.
            perda_com_peso = (perda_bruta * lote_w).mean()
            
            # Backward: O Juiz pega a prova errada e rastreia qual neurônio causou o erro
            # (feito fora do autocast por segurança matemática)
            perda_com_peso.backward()
            
            # Step: O professor aperta ou afrouxa as conexões dos neurônios culpados
            otimizador.step()

            # Apenas rastreamos a perda (Loss). Removemos o cálculo de Sigmoid/Acurácia 
            # daqui de dentro para não gastar poder de CPU à toa no loop quente!
            perda_acumulada += perda_com_peso.item()
            total_amostras += lote_y.size(0)
            
            t_comp = time.time() - t_comp_start
            
            # Printa o progresso e loga a performance
            if (batch_idx + 1) % 100 == 0:
                ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                print(f"   ⏳ Época {epoca+1} - Progresso: Lote {batch_idx+1}/{len(carregador_treino)} ({(batch_idx+1)/len(carregador_treino)*100:.1f}%)")
                logging.info(f"[RESUMO Lote {batch_idx+1}] Tempo Total IA: {t_comp:.4f}s | RAM Pico: {ram_mb:.1f} MB")
                
            if (batch_idx + 1) % 2000 == 0:
                gc.collect()
                
            t_dados_start = time.time()

        scheduler.step()
        perda_media = perda_acumulada / len(carregador_treino)
        print(f"📊 Fim da Época {epoca+1} | Loss do Treino: {perda_media:.4f}")
        
        # ------------------------------------------------------------------
        # FASE DE VALIDAÇÃO (A PROVA SURPRESA)
        # ------------------------------------------------------------------
        modelo.eval()
        perda_val_acumulada = 0.0
        acertos_val = 0
        total_val = 0
        
        with torch.no_grad():
            for lote_x, lote_y, lote_w in carregador_val:
                lote_x = lote_x.to(device, non_blocking=True).long()
                lote_y = lote_y.to(device, non_blocking=True)
                lote_w = lote_w.to(device, non_blocking=True)
                
                # Validação em Float32 nativo
                predicao = modelo(lote_x).view_as(lote_y)
                perda_bruta = criterio(predicao, lote_y)
                perda_com_peso = (perda_bruta * lote_w).mean()
                
                perda_val_acumulada += perda_com_peso.item()
                classes = (torch.sigmoid(predicao) >= 0.5).float()
                acertos_val += (classes == lote_y).sum().item()
                total_val += lote_y.size(0)
                
        perda_val_media = perda_val_acumulada / len(carregador_val) if len(carregador_val) > 0 else 0.0
        acuracia_val = acertos_val / total_val if total_val > 0 else 0.0
        
        print("="*50)
        print(f"🏆 RESULTADOS DA ÉPOCA {epoca+1}:")
        print(f"📉 Loss Treino: {perda_media:.4f} | 📉 Loss Validação: {perda_val_media:.4f}")
        print(f"🎯 Acurácia na Prova Surpresa: {acuracia_val*100:.2f}%")
        print("="*50)
        
        # O Salva-Vidas: Guarda um backup a cada Época caso falte energia.
        checkpoint_path = f"pesos/pesos_moderador_ep{epoca+1}.pth"
        torch.save(modelo.state_dict(), checkpoint_path)
        print(f"   💾 Checkpoint salvo em: {checkpoint_path}\n")
        gc.collect()

    # Treino Finalizado. Colocamos o modelo em modo 'Operário Frio' (Desliga o Dropout).
    modelo.eval()
    torch.save(modelo.state_dict(), "pesos/pesos_moderador.pth")
    print("🎯 Rede treinada com sucesso! Pesos finais salvos em 'pesos/pesos_moderador.pth'")

if __name__ == "__main__":
    treinar_moderador()
