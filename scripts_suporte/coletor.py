import os
import csv
import sqlite3
import datetime

def coletar_amostras():
    print("=" * 60)
    print(" 📦 RBooster - Coletor e Extrator de Amostras do Dataset")
    print("=" * 60)
    
    db_path = "banco/dataset.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Erro: O banco de dados '{db_path}' não foi encontrado!")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Mapeia todas as origens disponíveis no banco
    print("🔍 Mapeando origens e volumes no banco de dados...")
    cur.execute("SELECT origem, COUNT(*) FROM frases GROUP BY origem ORDER BY COUNT(*) DESC")
    origens_info = cur.fetchall()

    if not origens_info:
        print("⚠️ O banco de dados está vazio!")
        conn.close()
        return

    total_geral = sum(qtd for _, qtd in origens_info)

    print("\n📊 Origens encontradas:")
    print(f"   [0] 🌐 TODAS AS ORIGENS (Total: {total_geral:,} frases)")
    for i, (origem, qtd) in enumerate(origens_info, start=1):
        print(f"   [{i}] 📁 {origem} ({qtd:,} frases)")

    # 2. Escolha da Origem
    print("-" * 60)
    opcao_origem = input(f"Selecione o número da origem desejada [0 a {len(origens_info)}]: ").strip()

    origem_selecionada = None
    if opcao_origem == "0" or not opcao_origem:
        origem_selecionada = "TODAS"
        nome_origem_filtro = None
    elif opcao_origem.isdigit() and 1 <= int(opcao_origem) <= len(origens_info):
        nome_origem_filtro = origens_info[int(opcao_origem) - 1][0]
        origem_selecionada = nome_origem_filtro
    else:
        print("❌ Opção inválida! Cancelando.")
        conn.close()
        return

    # 3. Escolha da Quantidade
    qtd_input = input("Quantas amostras aleatórias deseja coletar? (Padrão: 100): ").strip()
    qtd_amostras = 100
    if qtd_input.isdigit() and int(qtd_input) > 0:
        qtd_amostras = int(qtd_input)

    # 4. Escolha do Filtro de Label (Opcional)
    print("\nDeseja filtrar por classe (Label)?")
    print("   [0] 🎲 Ambos (Tóxicos e Seguros misturados)")
    print("   [1] 🟢 Apenas SEGURO (Label = 0)")
    print("   [2] 🔴 Apenas TÓXICO (Label = 1)")
    label_opcao = input("Escolha o filtro de label [0, 1, 2] (Padrão: 0): ").strip()

    filtro_label = None
    if label_opcao == "1":
        filtro_label = 0
    elif label_opcao == "2":
        filtro_label = 1

    # 5. Montagem da Query SQL
    print(f"\n⏳ Buscando {qtd_amostras} amostras aleatórias de '{origem_selecionada}'...")
    
    clausulas = []
    parametros = []

    if nome_origem_filtro:
        clausulas.append("origem = ?")
        parametros.append(nome_origem_filtro)

    if filtro_label is not None:
        clausulas.append("label = ?")
        parametros.append(filtro_label)

    where_sql = f"WHERE {' AND '.join(clausulas)}" if clausulas else ""
    parametros.append(qtd_amostras)

    query = f"""
        SELECT rowid, text, label, origem 
        FROM frases 
        {where_sql} 
        ORDER BY RANDOM() 
        LIMIT ?
    """

    cur.execute(query, parametros)
    amostras = cur.fetchall()
    conn.close()

    if not amostras:
        print("⚠️ Nenhuma amostra encontrada com esses critérios.")
        return

    # 6. Criação da pasta e gravação do arquivo CSV
    pasta_destino = "amostras"
    os.makedirs(pasta_destino, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_label = "" if filtro_label is None else f"_label{filtro_label}"
    tag_origem = origem_selecionada.lower().replace(" ", "_")
    nome_arquivo = f"amostra_{tag_origem}{tag_label}_{len(amostras)}_{timestamp}.csv"
    caminho_csv = os.path.join(pasta_destino, nome_arquivo)

    # utf-8-sig permite que Excel e LibreOffice abram com acentos e emojis perfeitos
    with open(caminho_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "text", "label", "origem"])
        for row in amostras:
            writer.writerow(row)

    print(f"✅ {len(amostras)} amostras coletadas com sucesso!")
    print(f"📁 Arquivo CSV salvo em: \033[1;32m{caminho_csv}\033[0m")
    
    # 7. Prévia na tela
    print("\n👀 Prévia das primeiras 5 amostras:")
    print("-" * 60)
    for idx, (_, txt, lbl, orig) in enumerate(amostras[:5], 1):
        status = "🔴 Tóxico" if lbl == 1 else "🟢 Seguro"
        txt_curto = txt if len(txt) <= 80 else txt[:77] + "..."
        print(f" {idx}. [{status} | {orig}] {txt_curto}")
    print("-" * 60)

    # 8. Prompt enxuto pronto para copiar (para usar com o CSV anexado)
    print("\n" + "=" * 70)
    print("📋 COPIE ESTA INSTRUÇÃO E COLE NO CHAT JUNTO COM O CSV ANEXADO:")
    print("=" * 70)
    print("=" * 70)
    print("=" * 70)

    prompt_chat = f"""[CONTEXTO: AUDITORIA DE SEGURANÇA E DATASET DE MODERAÇÃO DE CONTEÚDO]
Você atua como um Especialista Sênior em Moderação de Conteúdo e Alinhamento Ético de IA.
Estou anexando um arquivo CSV contendo {len(amostras)} amostras aleatórias do meu banco de dados (Origem: {origem_selecionada}).

O arquivo possui as colunas: id, text, label e origem.
- Label 0 = SEGURO (Frase inofensiva, discordância educada, gíria do dia a dia, elogio, trabalho, amizade ou identidade).
- Label 1 = TÓXICO (Ofensa real, discurso de ódio, assédio, preconceito explícito/velado, ameaça ou cyberbullying).

SUA MISSÃO:
1. Analise as frases do CSV e aponte os ERROS DE ROTULAGEM:
   - Falsos Positivos: frases inofensivas/amigáveis que foram marcadas como 1 por engano.
   - Falsos Negativos: ofensas reais ou discursos de ódio que passaram como 0.
2. Para cada erro encontrado, liste o ID, o texto e qual deveria ser o label correto com uma breve justificativa.
3. No final, dê uma nota de 0 a 10 para a qualidade geral deste lote e aponte os principais padrões de viés ou ruído que você identificou."""

    print(prompt_chat)
    print("=" * 70)
    print(f"📎 Anexe o arquivo: \033[1;33m{caminho_csv}\033[0m")
    

if __name__ == "__main__":
    coletar_amostras()
