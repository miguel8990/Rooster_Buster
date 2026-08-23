import torch
import json
import os
import sqlite3
import random
import time
from modelo import ModeradorCNN
from openai import OpenAI

def clube_da_luta():
    print("========================================")
    print(" ⚔️ RBooster - Clube da Luta (RLAIF API) ⚔️")
    print("========================================")
    
    caminho_csv = "dados/dados_sinteticos.csv"
    
    # --- 1. CARREGA O MODELO APRENDIZ (CNN) ---
    with open("vocabulario.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    cnn = ModeradorCNN(vocab_size=len(vocab), embedding_dim=32, num_filtros=64)
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True)
    state_dict_limpo = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    cnn.load_state_dict(state_dict_limpo)
    cnn.eval()
    print("✅ Aprendiz (CNN) posicionado no ringue.")

    # --- 2. CARREGA O MODELO MESTRE (API DA TOGETHER AI) ---
    def ler_env(caminho):
        envs = {}
        if os.path.exists(caminho):
            with open(caminho, 'r') as f:
                for linha in f:
                    if '=' in linha and not linha.strip().startswith('#'):
                        chave, valor = linha.split('=', 1)
                        envs[chave.strip()] = valor.strip().strip("'").strip('"')
        return envs

    env_local = ler_env(".env")
    caminho_real = env_local.get("CAMINHO_API_KEY")
    modelo_mestre = env_local.get("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    # Se o caminho veio faltando a barra do root (ex: home/miguel), nós arrumamos
    if caminho_real and not caminho_real.startswith('/'):
        caminho_real = '/' + caminho_real

    env_global = ler_env(caminho_real) if caminho_real else {}
    
    # Busca pela chave configurada (OPENAI_API_KEY, que é usada pela Together também)
    chave_api = env_global.get("OPENAI_API_KEY") or env_global.get("TOGETHER_API_KEY") or env_global.get("API_KEY")
    
    if not chave_api:
        print(f"\n❌ ERRO: Chave 'OPENAI_API_KEY' não encontrada no arquivo {caminho_real}!")
        return
        
    mestre = OpenAI(
        api_key=chave_api,
        base_url="https://api.together.xyz/v1"
    )
    
    print(f"✅ Mestre API ({modelo_mestre} via Together AI) entrou no ringue.")

    cenarios = [
        # Cenários de Ódio Camuflado e Cyberbullying
        "Frase com ofensa camuflada trocando letras por números (ex: l1x0, c0varde, m3rda).",
        "Assédio moral disfarçado de feedback construtivo no ambiente de trabalho.",
        "Frase de cyberbullying contra um adolescente na escola.",
        "Ofensa direcionada à aparência física de alguém (body shaming) sem usar palavrões.",
        "Ameaça velada de agressão física na saída da escola ou do trabalho.",
        "Discurso de ódio atacando estrangeiros ou imigrantes trabalhando no país.",
        "Xingamento pesado usando gírias regionais brasileiras.",
        "Humilhação direcionada a alguém por não ter dinheiro ou ser de classe baixa.",
        "Comentário tóxico em partida de videogame mandando o jogador desinstalar o jogo ou se matar.",
        "Frase machista disfarçada de elogio ('você é inteligente para uma mulher').",
        
        # Cenários de Viés e Minorias (Testes de Falso Positivo e Negativo)
        "Frase perfeitamente normal que usa a palavra 'mulher' ou 'feminismo' de forma respeitosa.",
        "Comentário agressivo de ódio religioso (intolerância contra qualquer religião).",
        "Comentário respeitoso de exaltação à cultura afro-brasileira ou pessoas negras.",
        "Frase extremamente tóxica e racista sobre minorias.",
        "Notícia jornalística neutra falando sobre a comunidade LGBTQIA+.",
        "Ataque homofóbico agressivo usando palavras pejorativas fortes.",
        "Defesa apaixonada dos direitos dos povos indígenas e demarcação de terras.",
        "Insulto capacitista contra uma pessoa com deficiência intelectual ou física.",
        "Frase normal sobre obesidade do ponto de vista médico ou de aceitação (body positivity).",
        
        # Cenários Comuns do Cotidiano (Inocentes)
        "Frase do cotidiano com gírias normais de amizade e positividade (ex: tamo junto, valeu).",
        "Reclamação de um cliente insatisfeito com um produto, mas sem usar xingamentos.",
        "Um desabafo triste de alguém que teve um dia ruim no trabalho.",
        "Frase de motivação de academia ('tá pago', 'foco').",
        "Comentário fofo sobre animais de estimação.",
        "Debate político acalorado, porém respeitoso e sem ataques pessoais.",
        "Dúvida sincera de um iniciante pedindo ajuda em um fórum.",
        "Mensagem de bom dia de grupo de família no WhatsApp com emojis.",
        "Discussão sobre uma receita de comida ou restaurante.",
        
        # Pegadinhas Linguísticas (Avançado)
        "Frase contendo a palavra 'matar' em um contexto inofensivo (ex: 'vou matar a saudade', 'matar a sede').",
        "Frase usando palavras que são insultos em outros contextos, mas inofensivas aqui (ex: 'que bosta de chuva', 'droga de trânsito').",
        "Elogio agressivo (ex: 'você é foda pra caralho', 'seu vídeo tá pica').",
        "Ironia pesada onde o usuário não xinga, mas destrói o alvo psicologicamente.",
        
        # Cenários Específicos do Nicho: Rede Social Sobrenatural / Paranormal
        "Relato intenso e assustador de uma experiência com fantasmas ou demônios, mas sem ofender nenhum usuário.",
        "Cético chamando os membros do fórum de 'esquizofrênicos', 'burros' ou 'idiotas' por acreditarem em OVNIs ou espíritos.",
        "Membro fanático lançando uma maldição pesada ou ameaçando a vida espiritual de um cético que duvidou de seu post.",
        "Debate extremamente acalorado sobre a veracidade de um vídeo de poltergeist, mas focado no vídeo e mantendo o respeito mútuo.",
        "Usuário cético apontando educadamente que um vídeo de fantasma é apenas edição gráfica (CGI) ou pareidolia.",
        "Pessoa mandando outro usuário 'ir pro inferno' ou 'queimar no fogo' por discordar de sua postagem sobre bruxaria.",
        "Descrição bizarra e gráfica de um ritual mágico ou de ocultismo que é puramente informativo ou relato, sem incitar ódio real.",
        "Xingamento camuflado usando termos do nicho paranormal (ex: 'seu demônio d3sgraçado', 'assombração de m3rda', 'vampiro inútil').",
        "Usuário atacando diretamente a religião cristã ou matriz africana de alguém durante uma discussão sobre possessões.",
        "Desabafo de alguém que sofre de paralisia do sono ou vê vultos, pedindo ajuda à comunidade de forma respeitosa."
    ]

    print("\nDING DING DING! O Clube da Luta está pronto.")
    entrada_rounds = input("Quantos rounds autônomos deseja executar? (Pressione Enter para infinito, ou ex: 100): ").strip()
    
    limite_rounds = 0
    if entrada_rounds.isdigit():
        limite_rounds = int(entrada_rounds)
        print(f"🔥 O limite foi definido para {limite_rounds} rounds. Ele vai parar sozinho.")
    else:
        print("🔥 Modo Infinito ativado. (Pressione Ctrl+C a qualquer momento para parar com segurança).")
        
    print("")
    rounds = 1
    erros_cnn = 0
    
    try:
        while True:
            if limite_rounds > 0 and rounds > limite_rounds:
                print(f"\n🏁 Limite de {limite_rounds} rounds alcançado! Finalizando o treino com segurança...")
                break
                
            if random.random() < 0.50:
                # 50% de chance: O próprio LLM inventa um cenário inédito (Self-Instruct)
                prompt_criativo = """Você é um Engenheiro de Segurança de Redes Sociais. 
Sua tarefa é inventar UM cenário altamente específico e criativo para testar um filtro de toxicidade (focado no cotidiano, cyberbullying ou no nicho paranormal/sobrenatural).
O cenário pode ser de ódio explícito, ódio camuflado, ameaça velada, ironia, falso positivo (frase limpa mas confusa) ou debates acalorados entre céticos e crentes em demônios/fantasmas/ETs.
Responda APENAS com a descrição do cenário em 1 ou 2 frases, sem aspas e sem conversa extra."""
                
                try:
                    resp_tema = mestre.chat.completions.create(
                        model=modelo_mestre,
                        messages=[{"role": "user", "content": prompt_criativo}],
                        max_tokens=80,
                        temperature=0.9, # Mais temperatura = mais criatividade
                        extra_body={"reasoning": {"enabled": False}}
                    )
                    tema = "CENÁRIO INÉDITO CRIADO POR VOCÊ:\n" + resp_tema.choices[0].message.content.strip()
                    tipo_tema = "🧠 Dinâmico (IA)"
                except Exception:
                    tema = random.choice(cenarios)
                    tipo_tema = "📜 Fixo (Lista)"
            else:
                # 50% de chance: Pega da nossa lista com curadoria manual
                tema = random.choice(cenarios)
                tipo_tema = "📜 Fixo (Lista)"
                
            prompt = f"""Como um avaliador mestre de toxicidade, gere EXATAMENTE UMA frase curta (máximo 80 letras) seguindo esta diretriz:
{tema}
Em seguida, avalie se ela é Tóxica (1) ou Segura (0).
Responda APENAS neste formato rígido, sem aspas e sem explicações extras:
FRASE GERADA|ROTULO
Exemplo:
seu bolo ficou horrivel|0"""

            try:
                t0_api = time.time()
                resposta = mestre.chat.completions.create(
                    model=modelo_mestre,
                    messages=[
                        {"role": "system", "content": "Você é um gerador estrito de CSV no formato texto|label"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.95,
                    extra_body={"reasoning": {"enabled": False}}
                )
                t1_api = time.time()
                latencia_mestre = (t1_api - t0_api) * 1000  # em ms
                
                
                saida = resposta.choices[0].message.content.strip()
                
                # Varre as linhas procurando o CSV caso o LLM tente puxar papo
                linha_valida = None
                for linha in saida.split('\n'):
                    if '|' in linha:
                        linha_valida = linha.strip()
                        break
                        
                if not linha_valida:
                    print(f"⚠️ Mestre gaguejou (Formato inválido sem '|' ou cortado): {saida}")
                    rounds += 1
                    continue
                    
                texto, rotulo_str = linha_valida.split('|', 1)
                texto = texto.strip().replace('"', '') # Limpa aspas intrometidas
                rotulo_mestre = int(rotulo_str.strip())
                
            except Exception as e:
                print(f"⚠️ Erro ao processar o golpe do Mestre: {e}")
                rounds += 1
                continue
                
            # --- O APRENDIZ TENTA ADIVINHAR ---
            sequencia = [vocab.get(char, vocab["<UNK>"]) for char in texto.lower()]
            if len(sequencia) > 300:
                sequencia = sequencia[:300]
            else:
                sequencia = sequencia + [vocab["<PAD>"]] * (300 - len(sequencia))
                
            tensor = torch.tensor([sequencia], dtype=torch.long)
            t0_cnn = time.time()
            with torch.no_grad():
                pred = torch.sigmoid(cnn(tensor)).item()
            latencia_cnn = (time.time() - t0_cnn) * 1000
                
            rotulo_aprendiz = 1 if pred >= 0.5 else 0
            
            print(f"🥊 Round {rounds} [{tipo_tema}]: '{texto}'")
            print(f"   Mestre diz: {rotulo_mestre} (API: {latencia_mestre:.0f}ms) | Aprendiz diz: {rotulo_aprendiz} ({(pred*100):.1f}%, CPU local: {latencia_cnn:.2f}ms)")
            
            # Salva 100% dos dados gerados no SQLite (Reforço Positivo + Correção de Erros)
            try:
                conn_db = sqlite3.connect('banco/dataset.db')
                cursor = conn_db.cursor()
                cursor.execute(
                    "INSERT INTO frases (text, label, origem) VALUES (?, ?, ?)",
                    (texto, rotulo_mestre, 'sintetico_rlaif')
                )
                conn_db.commit()
                conn_db.close()
            except Exception as db_err:
                print(f"⚠️ Erro ao salvar no banco: {db_err}")

            if rotulo_mestre == rotulo_aprendiz:
                print("   ✨ Aprendiz esquivou! (Dado guardado no BD para reforçar a memória).")
            else:
                erros_cnn += 1
                print("   🩸 APRENDIZ ERROU! (Nova vacina injetada no BD com sucesso).")
                
            print("-" * 50)
            rounds += 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Luta interrompida pelo árbitro (Você).")
        
    print(f"📊 RESUMO DA SESSÃO:")
    print(f"   Rounds lutados: {rounds - 1}")
    print(f"   Novas Vacinas geradas (Erros do Aprendiz): {erros_cnn}")
    print("   Os dados já estão salvos e seguros no banco 'banco/dataset.db'.")
    print("   Rode 'python3 src/treinador.py' para a CNN aprender com os próprios erros!\n")

if __name__ == "__main__":
    clube_da_luta()
