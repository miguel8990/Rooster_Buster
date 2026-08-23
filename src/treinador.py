import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import sqlite3
import json
import os
from modelo import ModeradorCNN, contar_parametros

# Força o PyTorch a usar todas as 16 Threads do seu Ryzen 7 5700G!
torch.set_num_threads(16)

MAX_LEN = 300

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Preguiçoso (Lazy): Nunca carrega tudo na RAM.
# Mantém só os rowids em memória (~32MB para 4M linhas) e busca cada
# frase no SQLite apenas quando o DataLoader pede o lote.
# ─────────────────────────────────────────────────────────────────────────────
class SQLiteDataset(Dataset):
    def __init__(self, db_path, vocab, max_len=MAX_LEN):
        self.db_path = db_path
        self.vocab = vocab
        self.max_len = max_len

        # Lê APENAS os rowids e labels — isso é leve e cabe na RAM tranquilo
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT rowid, label FROM frases WHERE CAST(label AS REAL) IS NOT NULL")
        rows = cur.fetchall()
        conn.close()

        # Filtra lixo de cabeçalhos que viraram None ou string
        self.rowids = []
        self.labels = []
        for r in rows:
            try:
                self.labels.append(float(r[1]))
                self.rowids.append(r[0])
            except (TypeError, ValueError):
                pass

        print(f"📦 Dataset com {len(self.rowids)} amostras indexadas (RAM segura).")

    def __len__(self):
        return len(self.rowids)

    def __getitem__(self, idx):
        rowid = self.rowids[idx]
        label = self.labels[idx]

        # Abre uma conexão por item — SQLite suporta isso com check_same_thread=False
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT text FROM frases WHERE rowid = ?", (rowid,))
        row = cur.fetchone()
        conn.close()

        texto = str(row[0]).lower() if row else ""
        seq = [self.vocab.get(c, self.vocab["<UNK>"]) for c in texto]

        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [self.vocab["<PAD>"]] * (self.max_len - len(seq))

        return torch.tensor(seq, dtype=torch.long), torch.tensor([label], dtype=torch.float32)


def construir_vocabulario(db_path):
    print("🔤 Construindo vocabulário (leitura em chunks de 100k)...")
    caracteres_unicos = set()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    offset = 0
    chunk = 100_000
    while True:
        cur.execute("SELECT text FROM frases LIMIT ? OFFSET ?", (chunk, offset))
        rows = cur.fetchall()
        if not rows:
            break
        for (txt,) in rows:
            for char in str(txt).lower():
                caracteres_unicos.add(char)
        offset += chunk
        print(f"   ⏳ {offset} textos escaneados para vocab...")

    conn.close()

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, char in enumerate(sorted(caracteres_unicos), start=2):
        vocab[char] = i

    print(f"✅ Vocabulário: {len(vocab)} caracteres únicos.")
    with open("vocabulario.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab


def calcular_pos_weight(db_path):
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
    conn.close()
    limpos = total - toxicos
    peso = limpos / max(toxicos, 1)
    print(f"⚖️  pos_weight = {peso:.2f}x  ({limpos} seguros / {toxicos} tóxicos)")
    return torch.tensor([peso], dtype=torch.float32)


def treinar_moderador():
    db_path = "banco/dataset.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Banco não encontrado: {db_path}. Rode a migração primeiro!")

    vocab = construir_vocabulario(db_path)
    peso_positivo = calcular_pos_weight(db_path)

    dataset = SQLiteDataset(db_path, vocab)

    # num_workers=0: cada __getitem__ abre/fecha sua conexão SQLite
    # Sem multiprocessing para evitar conflito de fork com torch.compile
    carregador = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    modelo = ModeradorCNN(vocab_size=len(vocab), embedding_dim=32, num_filtros=64)
    contar_parametros(modelo)

    print("⚙️  Compilando a CNN em C++ nativo (pode levar alguns segundos)...")
    modelo = torch.compile(modelo)

    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivo)
    otimizador = optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(otimizador, T_max=10, eta_min=1e-5)

    epocas = 10
    print("🥊 Iniciando a Caçada por Padrões Tóxicos!\n")

    for epoca in range(epocas):
        modelo.train()
        perda_acumulada = 0.0
        acertos = 0
        total_amostras = 0

        for lote_x, lote_y in carregador:
            otimizador.zero_grad(set_to_none=True)
            predicao_bruta = modelo(lote_x)
            perda = criterio(predicao_bruta, lote_y)
            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item()
            classes = (torch.sigmoid(predicao_bruta) >= 0.5).float()
            acertos += (classes == lote_y).sum().item()
            total_amostras += lote_y.size(0)

        scheduler.step()
        perda_media = perda_acumulada / len(carregador)
        acuracia = 100.0 * acertos / total_amostras
        lr_atual = scheduler.get_last_lr()[0]
        print(f"Época [{epoca+1:02d}/{epocas}] | Loss: {perda_media:.4f} | Acurácia: {acuracia:.2f}% | LR: {lr_atual:.6f}")

    modelo.eval()
    os.makedirs("pesos", exist_ok=True)
    torch.save(modelo.state_dict(), "pesos/pesos_moderador.pth")
    print("\n✅ Rede treinada! Pesos salvos em 'pesos/pesos_moderador.pth'")

if __name__ == "__main__":
    treinar_moderador()
