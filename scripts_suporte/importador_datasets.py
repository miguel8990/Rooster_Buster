import os
import re
import csv
import html
import sqlite3

DB_PATH = "banco/dataset.db"
DADOS_DIR = "dados"

def limpar_texto(texto):
    """
    Higieniza o texto: remove links, menções de scraping, hashtags,
    tags HTML e espaços múltiplos, preservando 100% de emojis e acentuação PT-BR.
    """
    if not texto:
        return ""
    
    # Decodifica HTML entities (&amp; -> &, &quot; -> ", etc.)
    t = html.unescape(str(texto))
    
    # Remove URLs completas (http, https, www)
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    
    # Remove menções e hashtags completas (@user, #hashtag, etc.)
    t = re.sub(r'@[A-Za-z0-9_]+', '', t)
    t = re.sub(r'#\w+', '', t)
    t = re.sub(r'(?i)\b(hashtag|user)\b', '', t)
    t = re.sub(r'[@#]+', '', t)
    
    # Remove quebras de linha excessivas e caracteres de controle
    t = re.sub(r'[\r\n\t]+', ' ', t)
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', t)
    
    # Normaliza espaços múltiplos
    t = re.sub(r'\s+', ' ', t).strip()
    
    return t

def painel_interativo_humano(texto, origem, info_extra):
    """
    Painel interativo no terminal para o usuário decidir sobre casos não-conformes ou duvidosos.
    """
    print("\n" + "=" * 70)
    print(" ⚠️  PAINEL INTERATIVO DE CONFORMIDADE - CASO DUVIDOSO")
    print("=" * 70)
    print(f" 📁 Origem/Arquivo: \033[1;36m{origem}\033[0m")
    print(f" ℹ️ Detalhes brutos: {info_extra}")
    print(f" 📏 Tamanho: {len(texto)} caracteres")
    print("-" * 70)
    print(f" 💬 Texto:\n \"\033[1;33m{texto}\033[0m\"")
    print("-" * 70)
    print(" O que você deseja fazer com esta frase?")
    print("   [0] Salvar como 🟢 SEGURO (0.0)")
    print("   [1] Salvar como 🔴 TÓXICO (1.0)")
    print("   [e] ✏️  Editar texto e definir label")
    print("   [s] ⏭️  Pular esta frase (Descartar)")
    print("   [a] ⚡ Aceitar decisão automática para as próximas deste arquivo")
    
    while True:
        escolha = input("👉 Escolha uma opção [0, 1, e, s, a]: ").strip().lower()
        if escolha == '0':
            return texto, 0.0, False
        elif escolha == '1':
            return texto, 1.0, False
        elif escolha == 's':
            return None, None, False
        elif escolha == 'a':
            return texto, None, True
        elif escolha == 'e':
            novo_texto = input("Digite o novo texto corrigido: ").strip()
            if not novo_texto:
                print("Texto vazio! Tente novamente.")
                continue
            lbl = input("Qual o label? [0 = Seguro, 1 = Tóxico]: ").strip()
            if lbl in ['0', '1']:
                return novo_texto, float(lbl), False
            print("Label inválido!")
        else:
            print("Opção inválida! Escolha 0, 1, e, s ou a.")

def inserir_frase(conn, texto, label, origem):
    """
    Insere no SQLite com regra estrita de não-repetição (UNIQUE) e tamanho [3, 300].
    """
    if not texto or len(texto) < 3 or len(texto) > 300:
        return 0
    
    if label not in [0.0, 1.0]:
        return 0
    
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO frases (text, label, origem) VALUES (?, ?, ?)", (texto, label, origem))
    return cur.rowcount

def parse_label_generico(valor):
    """
    Tenta deduzir se um valor genérico de label é 1.0 (Tóxico) ou 0.0 (Seguro).
    Retorna float(0.0), float(1.0) ou None (se ambíguo).
    """
    if valor is None:
        return None
    
    v_str = str(valor).strip().lower()
    
    # Valores óbvios de Tóxico
    if v_str in ['1', '1.0', 'true', 'toxic', 'toxico', 'hate', 'hateful', 'off', 'offensive', 'ofensivo', 'sim', 's', 'yes', 'y']:
        return 1.0
    
    # Valores óbvios de Seguro
    if v_str in ['0', '0.0', 'false', 'safe', 'seguro', 'non-toxic', 'non_toxic', 'nontoxic', 'not', 'clean', 'neutro', 'neutral', 'nao', 'n', 'no']:
        return 0.0
    
    # Tenta conversão numérica
    try:
        f = float(v_str)
        if f >= 0.5:
            return 1.0
        else:
            return 0.0
    except ValueError:
        return None

def processar_arquivo_csv(caminho_csv, conn, auto_mode=False):
    """
    Processador Universal: detecta automaticamente o formato (se for TuPyE, ToLD, HateCheck, OLID)
    ou aplica heurísticas inteligentes para qualquer arquivo CSV arbitrário que o usuário adicionar!
    """
    nome_arquivo = os.path.basename(caminho_csv)
    origem_padrao = os.path.splitext(nome_arquivo)[0].lower()
    
    print(f"\n📂 Analisando arquivo: \033[1;36m{nome_arquivo}\033[0m ...")
    
    with open(caminho_csv, 'r', encoding='utf-8', errors='ignore') as f:
        # Detecta separador (, ou ;)
        amostra = f.read(4096)
        f.seek(0)
        sep = ';' if amostra.count(';') > amostra.count(',') else ','
        
        reader = csv.DictReader(f, delimiter=sep)
        colunas = reader.fieldnames
        if not colunas:
            print(f"❌ Não foi possível ler as colunas de {nome_arquivo}!")
            return 0, 0, 0
        
        colunas_lower = {c.lower().strip(): c for c in colunas}
        
        # -------------------------------------------------------------
        # 1. AUTO-DETECÇÃO DE DATASETS CONHECIDOS
        # -------------------------------------------------------------
        tipo = "generico"
        col_texto = None
        col_label = None
        
        if "hate" in colunas_lower and "aggressive" in colunas_lower:
            tipo = "tupye"
            print("   🔍 Formato Reconhecido: \033[1;32mTuPyE Dataset\033[0m")
        elif "homophobia" in colunas_lower and "insult" in colunas_lower and "racism" in colunas_lower:
            tipo = "told_br"
            print("   🔍 Formato Reconhecido: \033[1;32mToLD-BR Dataset\033[0m")
        elif "label_gold" in colunas_lower and "test_case" in colunas_lower:
            tipo = "hatecheck"
            print("   🔍 Formato Reconhecido: \033[1;32mHateCheck-PT Dataset\033[0m")
        elif "is_offensive" in colunas_lower and "text" in colunas_lower:
            tipo = "olid_br"
            print("   🔍 Formato Reconhecido: \033[1;32mOLID-BR Dataset\033[0m")
        else:
            # Dataset desconhecido / arbitrário
            print("   🔍 Formato: \033[1;33mDataset Personalizado / Genérico\033[0m")
            
            # Caça coluna de texto
            candidatos_texto = ['text', 'texto', 'frase', 'comment', 'comentario', 'tweet', 'content', 'mensagem', 'msg', 'post', 'body']
            for cand in candidatos_texto:
                if cand in colunas_lower:
                    col_texto = colunas_lower[cand]
                    break
            
            # Caça coluna de label
            candidatos_label = ['label', 'rotulo', 'target', 'is_toxic', 'toxic', 'toxico', 'hate', 'offensive', 'classe', 'class', 'toxicidade', 'categoria']
            for cand in candidatos_label:
                if cand in colunas_lower:
                    col_label = colunas_lower[cand]
                    break
            
            # Se não encontrou automaticamente, pergunta ao usuário
            if not col_texto:
                print(f"   Colunas disponíveis: {list(colunas)}")
                col_texto = input(f"   👉 Digite o nome da coluna de TEXTO em '{nome_arquivo}': ").strip()
            
            if not col_label:
                print(f"   Colunas disponíveis: {list(colunas)}")
                col_label = input(f"   👉 Digite o nome da coluna de LABEL (0/1) em '{nome_arquivo}': ").strip()
            
            print(f"   ⚙️ Configuração: Texto='{col_texto}' | Label='{col_label}'")
        
        # -------------------------------------------------------------
        # 2. PROCESSAMENTO LINHA A LINHA
        # -------------------------------------------------------------
        total_lidos = 0
        inseridos = 0
        duplicados = 0
        
        f.seek(0)
        reader = csv.DictReader(f, delimiter=sep)
        
        for row in reader:
            total_lidos += 1
            lbl = None
            raw_text = ""
            
            if tipo == "tupye":
                raw_text = row.get("text", "")
                hate = str(row.get("hate", "")).strip()
                aggressive = str(row.get("aggressive", "")).strip()
                if hate in ['1', '1.0', 1] or aggressive in ['1', '1.0', 1]:
                    lbl = 1.0
                elif hate in ['0', '0.0', 0] and aggressive in ['0', '0.0', 0]:
                    lbl = 0.0
                else:
                    info = f"hate={hate}, aggressive={aggressive}"
            elif tipo == "told_br":
                raw_text = row.get("text", "")
                try:
                    scores = [
                        float(row.get("homophobia", 0) or 0),
                        float(row.get("obscene", 0) or 0),
                        float(row.get("insult", 0) or 0),
                        float(row.get("racism", 0) or 0),
                        float(row.get("misogyny", 0) or 0),
                        float(row.get("xenophobia", 0) or 0)
                    ]
                    soma_toxica = sum(scores)
                    if soma_toxica >= 1.0:
                        lbl = 1.0
                    elif soma_toxica == 0.0:
                        lbl = 0.0
                    info = f"Soma={soma_toxica:.1f}"
                except Exception:
                    lbl = 0.0
                    info = "Erro numérico"
            elif tipo == "hatecheck":
                raw_text = row.get("test_case", "")
                gold = str(row.get("label_gold", "")).strip().lower()
                if gold in ["hateful", "hate", "1"]:
                    lbl = 1.0
                elif gold in ["non-hateful", "non_hateful", "safe", "0"]:
                    lbl = 0.0
                info = f"label_gold={gold}"
            elif tipo == "olid_br":
                raw_text = row.get("text", "")
                is_off = str(row.get("is_offensive", "")).strip().upper()
                if is_off == "OFF":
                    lbl = 1.0
                elif is_off == "NOT":
                    lbl = 0.0
                info = f"is_offensive={is_off}"
            else:
                # Genérico
                raw_text = row.get(col_texto, "")
                val_label = row.get(col_label, None)
                lbl = parse_label_generico(val_label)
                info = f"{col_label}={val_label}"
                
            cleaned = limpar_texto(raw_text)
            
            # Se o label for desconhecido ou duvidoso
            if lbl is None:
                if not auto_mode:
                    cleaned, lbl, auto_mode = painel_interativo_humano(cleaned, nome_arquivo, info)
                else:
                    lbl = 0.0
                    
            if cleaned and lbl is not None:
                rc = inserir_frase(conn, cleaned, lbl, origem_padrao)
                if rc > 0:
                    inseridos += 1
                else:
                    duplicados += 1
                    
            if total_lidos % 5000 == 0:
                print(f"   ⏳ {nome_arquivo}: {total_lidos} linhas processadas...")
                conn.commit()
                
        conn.commit()
        print(f"✅ {nome_arquivo} Concluído! Lidos: {total_lidos:,} | Novos Inseridos: {inseridos:,} | Duplicados/Ignorados: {duplicados:,}")
        return total_lidos, inseridos, duplicados

# ==============================================================================
# MENU PRINCIPAL DINÂMICO
# ==============================================================================

def main():
    print("=" * 70)
    print(" 🛠️  RBooster - Tratador & Saneador Universal de Datasets (.CSV -> DB)")
    print("=" * 70)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Banco '{DB_PATH}' não encontrado!")
        return
    
    if not os.path.exists(DADOS_DIR):
        os.makedirs(DADOS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS frases (text TEXT UNIQUE, label REAL, origem TEXT)")
    
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM frases")
    total_antes = cur.fetchone()[0]
    print(f"📊 Total atual de frases no banco: \033[1;32m{total_antes:,} frases\033[0m\n")
    
    # Escaneia dinamicamente todos os .csv da pasta dados/
    arquivos_csv = sorted([f for f in os.listdir(DADOS_DIR) if f.lower().endswith('.csv')])
    
    if not arquivos_csv:
        print(f"⚠️ Nenhum arquivo .csv encontrado na pasta '{DADOS_DIR}/'.")
        print("Coloque seus arquivos .csv dentro da pasta 'dados/' e execute este script novamente!")
        conn.close()
        return
    
    print(f"📁 Arquivos .csv encontrados na pasta '{DADOS_DIR}/':")
    for idx, a in enumerate(arquivos_csv, 1):
        caminho_completo = os.path.join(DADOS_DIR, a)
        tam_mb = os.path.getsize(caminho_completo) / (1024 * 1024)
        print(f"   [{idx}] 📄 {a} ({tam_mb:.2f} MB)")
        
    print("\n   [0] 🚀 TRATAR E PROCESSAR TODOS OS ARQUIVOS LISTADOS ACIMA (Recomendado)")
    
    opcao = input(f"\n👉 Escolha uma opção [0 a {len(arquivos_csv)}, ou separados por vírgula como 1,2]: ").strip()
    
    modo_interativo = input("Deseja ativar o Painel Interativo para casos duvidosos? (s/n) [Padrão: s]: ").strip().lower()
    auto_mode = (modo_interativo == 'n')
    
    arquivos_para_processar = []
    if opcao == '0' or opcao == '':
        arquivos_para_processar = [os.path.join(DADOS_DIR, a) for a in arquivos_csv]
    else:
        partes = [p.strip() for p in opcao.split(',')]
        for p in partes:
            if p.isdigit():
                num = int(p)
                if 1 <= num <= len(arquivos_csv):
                    arquivos_para_processar.append(os.path.join(DADOS_DIR, arquivos_csv[num - 1]))
    
    if not arquivos_para_processar:
        print("Nenhum arquivo válido selecionado.")
        conn.close()
        return
        
    print("\n🚀 Iniciando esteira de saneamento e injeção...")
    for arq in arquivos_para_processar:
        processar_arquivo_csv(arq, conn, auto_mode)
        
    # Relatório final
    cur.execute("SELECT COUNT(*) FROM frases")
    total_depois = cur.fetchone()[0]
    novas_adicionadas = total_depois - total_antes
    
    print("\n" + "=" * 70)
    print(" 🎉 TRATAMENTO E SANEAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print(f"📈 Total antes: {total_antes:,} frases")
    print(f"🚀 Total agora no DB: \033[1;32m{total_depois:,} frases\033[0m (+{novas_adicionadas:,} novas frases limpas)")
    print("\n📊 Distribuição consolidada no banco:")
    cur.execute("SELECT origem, label, COUNT(*) FROM frases GROUP BY origem, label ORDER BY origem, label")
    for orig, lbl, cnt in cur.fetchall():
        tag_lbl = "🔴 Tóxico" if lbl == 1.0 else "🟢 Seguro"
        print(f"   - [{orig}] {tag_lbl}: {cnt:,}")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    main()
