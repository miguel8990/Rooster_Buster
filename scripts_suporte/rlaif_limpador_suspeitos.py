import os
import sys
import json
import time
import random
import sqlite3
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from modelo import ModeradorCNN
from openai import OpenAI

DB_PATH = "banco/dataset.db"
BATCH_SIZE = 15  # Envia 10 frases por chamada para economizar 90% dos tokens

def limpador_suspeitos():
    print("=" * 70)
    print(" 🚀⚡ RBooster - Re-Rotulador Ultrarrápido em Lote (10 Frases/Chamada)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. CARREGA O MODELO APRENDIZ (CNN)
    # -------------------------------------------------------------------------
    if not os.path.exists("vocabulario.json") or not os.path.exists("pesos/pesos_moderador.pth"):
        print("❌ Erro: 'vocabulario.json' ou 'pesos/pesos_moderador.pth' não encontrados!")
        return

    with open("vocabulario.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    cnn = ModeradorCNN(vocab_size=len(vocab), embedding_dim=256, num_filtros=512)
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True, map_location='cpu')
    state_dict_limpo = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    cnn.load_state_dict(state_dict_limpo)
    cnn.eval()
    print("✅ Aprendiz (CNN de 9 Lentes) pronto para validação cruzada.")

    # -------------------------------------------------------------------------
    # 2. CARREGA O MODELO MESTRE (API TOGETHER AI / DEEPSEEK)
    # -------------------------------------------------------------------------
    def ler_env(caminho):
        envs = {}
        if caminho and os.path.exists(caminho):
            with open(caminho, 'r') as f:
                for linha in f:
                    if '=' in linha and not linha.strip().startswith('#'):
                        chave, valor = linha.split('=', 1)
                        envs[chave.strip()] = valor.strip().strip("'").strip('"')
        return envs

    env_local = ler_env(".env")
    caminho_real = env_local.get("CAMINHO_API_KEY")
    modelo_mestre = env_local.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")

    if caminho_real and not caminho_real.startswith('/'):
        caminho_real = '/' + caminho_real

    env_global = ler_env(caminho_real) if caminho_real else {}
    chave_api = env_global.get("OPENAI_API_KEY") or env_global.get("TOGETHER_API_KEY") or env_global.get("API_KEY")

    if not chave_api:
        print(f"\n❌ ERRO: Chave de API não encontrada em {caminho_real} ou .env!")
        return

    mestre = OpenAI(
        api_key=chave_api,
        base_url="https://api.together.xyz/v1"
    )
    print(f"✅ Mestre API ({modelo_mestre}) conectado com sucesso.")

    # -------------------------------------------------------------------------
    # 3. VERIFICA O BANCO DE DADOS
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM frases WHERE origem = 'suspeito'")
    total_suspeitos = cur.fetchone()[0]
    conn.close()

    print(f"📊 Frases com origem='suspeito' restantes para re-rotular: \033[1;33m{total_suspeitos:,}\033[0m")
    if total_suspeitos == 0:
        print("🎉 Nenhuma frase suspeita restante no banco! Todas já foram re-rotuladas.")
        return

    entrada_chamadas = input(f"\nQuantas chamadas de API deseja realizar? (Ex: 10 chamadas = 100 frases | Enter para infinito): ").strip()
    limite_chamadas = int(entrada_chamadas) if entrada_chamadas.isdigit() else 0

    print(f"\n🚀 Iniciando esteira de re-rotulagem em lotes de {BATCH_SIZE} frases...")
    print(f"📡 Limite de Chamadas de API: \033[1;36m{limite_chamadas if limite_chamadas > 0 else 'ILIMITADO'}\033[0m")
    print("Pressione Ctrl+C a qualquer momento para pausar com segurança.\n")

    total_chamadas_api = 0
    total_frases_corrigidas = 0
    alterados_0 = 0
    alterados_1 = 0
    mantidos = 0

    def vetorizar(texto, max_len=300):
        unk_id = vocab.get('<UNK>', 1)
        pad_id = vocab.get('<PAD>', 0)
        v = [vocab.get(c.lower(), unk_id) for c in texto[:max_len]]
        if len(v) < max_len:
            v += [pad_id] * (max_len - len(v))
        return torch.tensor([v], dtype=torch.long)

    try:
        while True:
            # Trava Rígida de Chamadas de API
            if limite_chamadas > 0 and total_chamadas_api >= limite_chamadas:
                print(f"\n🛑 Limite de {limite_chamadas} chamadas de API atingido! Encerrando para proteger seus créditos.")
                break

            # Puxa 10 frases aleatórias de origem suspeito
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(f"SELECT text, label FROM frases WHERE origem = 'suspeito' ORDER BY RANDOM() LIMIT {BATCH_SIZE}")
            lote = cur.fetchall()
            conn.close()

            if not lote:
                print("\n🎉 Todas as frases suspeitas foram 100% re-rotuladas e saneadas!")
                break

            # Monta o prompt compacto em lote
            lista_formatada = "\n".join([f'{i+1}. "{item[0]}"' for i, item in enumerate(lote)])

            prompt = f"""[CONTEXTO: MODERAÇÃO DE CONTEÚDO E SEGURANÇA EM PORTUGUÊS DO BRASIL]
Você é um Especialista Sênior em Moderação de Conteúdo.
Avalie com rigor contextual as {len(lote)} frases reais da internet abaixo e determine o rótulo de cada uma:

DIRETRIZES DE ROTULAGEM:
- 1 (TÓXICO): Discurso de ódio, racismo, misoginia, homofobia, ameaças reais, cyberbullying, assédio, crueldade mórbida ou ofensas com disfarces/leetspeak (ex: lix0, m3rda, vagabundo imundo).
- 0 (SEGURO): Elogios com gírias ("o cara é um monstro na programação", "foda demais"), desabafos de dor/cansaço ("me odeio cólica"), torcida de esportes, piadas, defesa de minorias e palavrões inofensivos de espanto/alívio ("pqp que jogada linda").

LISTA DE FRASES:
{lista_formatada}

RETORNE ESTRITAMENTE APENAS UM ARRAY JSON CONTENDO OS {len(lote)} NÚMEROS (0 ou 1) NA MESMA ORDEM EXATA:
[0, 1, 0, 0, 1, 0, 1, 1, 0, 0]"""

            # 1. CHAMADA DA API
            try:
                total_chamadas_api += 1
                t0 = time.time()
                resposta = mestre.chat.completions.create(
                    model=modelo_mestre,
                    messages=[
                        {"role": "system", "content": "Você é um classificador de toxicidade em PT-BR. Saída estritamente em JSON Array de inteiros (0 ou 1)."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=80,  # Tokens mínimos de saída com folga para 15 inteiros
                    temperature=0.1,  # Temperatura baixa para máxima precisão e consistência
                    extra_body={"reasoning": {"enabled": False}}
                )
                tempo_api = time.time() - t0
                texto_resp = resposta.choices[0].message.content.strip()
            except Exception as e_api:
                print(f"\n🔥 [FALHA CATASTRÓFICA NA API] Chamada #{total_chamadas_api}: {e_api}")
                print("⏳ Ativando Cooldown de Segurança de 5 segundos para a API respirar...")
                time.sleep(5)
                continue

            # 2. PARSE DO ARRAY DE NÚMEROS
            try:
                if "```json" in texto_resp:
                    texto_resp = texto_resp.split("```json")[1].split("```")[0].strip()
                elif "```" in texto_resp:
                    texto_resp = texto_resp.split("```")[1].split("```")[0].strip()

                novos_rotulos = json.loads(texto_resp)
                if not isinstance(novos_rotulos, list) or len(novos_rotulos) != len(lote):
                    print(f"⚠️ [AVISO] IA retornou {len(novos_rotulos) if isinstance(novos_rotulos, list) else 0} números em vez de {len(lote)}. Pulando sem cooldown...")
                    continue
            except Exception as e_json:
                print(f"⚠️ [ERRO DE JSON] Resposta: '{texto_resp[:50]}...' ({e_json}). Pulando sem cooldown...")
                continue

            # 3. ATUALIZAÇÃO NO BANCO DE DADOS
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()

            limite_label = f"/{limite_chamadas}" if limite_chamadas > 0 else ""
            print("\n" + "─" * 70)
            print(f"📡 API #{total_chamadas_api}{limite_label} | ⏱️  Tempo: {tempo_api:.2f}s | 🔄 Re-Rotulando {len(lote)} Frases:")

            for idx, ((txt, lbl_antigo), novo_lbl_raw) in enumerate(zip(lote, novos_rotulos), 1):
                novo_lbl = float(novo_lbl_raw)

                # Previsão ao vivo da CNN
                with torch.no_grad():
                    tensor_in = vetorizar(txt)
                    logit = cnn(tensor_in).item()
                    prob_cnn = torch.sigmoid(torch.tensor(logit)).item()
                    lbl_cnn = 1.0 if prob_cnn >= 0.5 else 0.0

                # Atualiza no SQLite mudando origem para 'sintetico_rlaif'
                cur.execute("""
                    UPDATE frases 
                    SET label = ?, origem = 'sintetico_rlaif' 
                    WHERE text = ? AND origem = 'suspeito'
                """, (novo_lbl, txt))

                total_frases_corrigidas += 1
                if novo_lbl == 0.0 and lbl_antigo != 0.0:
                    alterados_0 += 1
                    status_mudanca = "\033[1;32m[1 ➔ 0 SEGURO]\033[0m"
                elif novo_lbl == 1.0 and lbl_antigo != 1.0:
                    alterados_1 += 1
                    status_mudanca = "\033[1;31m[0 ➔ 1 TÓXICO]\033[0m"
                else:
                    mantidos += 1
                    status_mudanca = f"[\033[1;30mMANTIDO {int(novo_lbl)}\033[0m]"

                tag_final = "🟢 0" if novo_lbl == 0.0 else "🔴 1"
                acerto_cnn = "✅" if lbl_cnn == novo_lbl else "🥊"

                print(f"   [{idx:02d}] {status_mudanca} {tag_final} | {acerto_cnn} CNN: {prob_cnn*100:4.1f}% | \"{txt[:60]}...\"")

            conn.commit()

            cur.execute("SELECT COUNT(*) FROM frases WHERE origem = 'suspeito'")
            restantes = cur.fetchone()[0]
            conn.close()

            print(f"💾 Lote gravado no banco! | 📉 Restam {restantes:,} frases suspeitas.")

    except KeyboardInterrupt:
        print("\n\n⏸️  Re-rotulagem pausada com sucesso pelo usuário!")

    print("\n" + "=" * 70)
    print(" 📊 RESUMO DA RE-ROTULAGEM EM LOTE")
    print("=" * 70)
    print(f"📡 Chamadas de API Realizadas: \033[1;33m{total_chamadas_api:,}\033[0m")
    print(f"🔄 Total de Frases Re-Rotuladas: \033[1;36m{total_frases_corrigidas:,}\033[0m")
    print(f"   - 🟢 Corrigidas para Seguro (1 ➔ 0): {alterados_0:,}")
    print(f"   - 🔴 Corrigidas para Tóxico (0 ➔ 1): {alterados_1:,}")
    print(f"   - ✅ Rótulos Confirmados e Mantidos: {mantidos:,}")
    print(f"💾 Banco de dados SQLite atualizado instantaneamente.")
    print("=" * 70)

if __name__ == "__main__":
    limpador_suspeitos()
