import torch
import json
import os
import sqlite3
import random
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
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
        
    cnn = ModeradorCNN(vocab_size=len(vocab), embedding_dim=256, num_filtros=512)
    state_dict = torch.load("pesos/pesos_moderador.pth", weights_only=True, map_location='cpu')
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

    cenarios_padrao = [
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
        "Frase que usam de palavras comuns como (preto, amarelo, franga etc) para disparar ofensas veladas.",
        "Ameaça de vazar dados pessoais, fotos privadas ou endereço (doxxing) em redes sociais.",
        "Frase passivo-agressiva de gaslighting psicológico ('ninguém te aguenta aqui', 'todo mundo finge que gosta de você').",
        "Comentário invasivo de assédio sexual ou insinuação não solicitada em postagem pública.",
        
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
        "Ameaças de agressão, morte ou violência de qualquer natureza direcionada ao individuo ou pessoas relacionadas a ele.",
        "Debate acalorado sobre cotas universitárias ou políticas públicas mantendo a civilidade.",
        "Comentário xenofóbico atacando sotaques e costumes de regiões específicas do Brasil (ex: Nordeste, Sul).",
        
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
        "Conversa amigável de grupo de condomínio sobre barulho ou vagas de garagem sem ofensas.",
        "Negociação em marketplace online (OLX / Mercado Livre) com dúvidas técnicas sobre um produto.",
        "Reclamação justa sobre atraso de voo, ônibus ou transporte público sem xingar os funcionários.",
        "Auto-depreciação cômica sobre cansaço ou falta de habilidade ('eu sou uma fraude cozinhando kkkk').",
        
        # Pegadinhas Linguísticas e Termos de Duplo Sentido (Avançado)
        "Frase contendo a palavra 'matar' em um contexto inofensivo (ex: 'vou matar a saudade', 'matar a sede').",
        "Frase usando palavras que são insultos em outros contextos, mas inofensivas aqui (ex: 'que bosta de chuva', 'droga de trânsito').",
        "Elogio agressivo (ex: 'você é foda pra caralho', 'seu vídeo tá pica', 'dar a pata').",
        "Ironia pesada onde o usuário não xinga, mas destrói o alvo psicologicamente.",
        "Expressão de surpresa ou admiração com palavrão de intensidade (ex: 'caralho mano que golaço absurdo', 'eita porra ficou lindo').",
        "Xingamento direcionado a objetos inanimados ou software (ex: 'computador desgraçado travou na hora do render', 'código de merda não compila').",
        "Desabafo sobre estudo ou estresse físico (ex: 'essa prova da faculdade me assassinou', 'tô morto de dor nas costas').",
        "Expressão de meme popular brasileiro com exagero cômico (ex: 'se você não gosta de coxinha você é meu inimigo').",
        
        # Cultura Gamer, Discord e Competição Online
        "Comunicação técnica e rápida de jogo competitivo (ex: 'vamos rushar e explodir a base B', 'dei headshot no suporte', 'dropei a bomba').",
        "Trash talk esportivo leve e saudável entre rivais (ex: 'GG fácil demais', 'meu time amassou o seu', 'chora não freguês').",
        "Ataque tóxico pesado em call de voz mandando o companheiro de equipe desinstalar o jogo ou se machucar.",
        "Discussão sobre balanceamento de personagens em fórum de jogos sem ofensa pessoal.",
        
        # Rivalidade Esportiva e Fandoms de Cultura Pop
        "Zoação sadia entre torcedores de futebol sobre o resultado do clássico no fim de semana.",
        "Crítica severa e apaixonada a um filme, roteiro ou final de série sem atacar os atores pessoalmente.",
        "Debate fervoroso sobre qual cantora ou banda é melhor sem incitar linchamento virtual.",
        
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

    cenarios_to_patch = [
        # Cenários Focados na Cura de Falsos Positivos (Desenviesamento e Inclusão)
        "Elogio e reconhecimento profissional legítimo para pessoas negras, mulheres ou pessoas LGBTQIA+ (ex: 'ele é negro e o melhor dev', 'ela é trans e manda muito').",
        "Frase cotidiana e afetuosa de amizade usando termos sensíveis (ex: 'meu melhor amigo é negro e gente boa', 'os amigos gays da faculdade').",
        "Mensagem inofensiva de rotina ou horários contendo a palavra amigo (ex: 'meu amigo disse que em 30 min vai estar aqui', 'fui com meu amigo no shopping').",
        "Exaltação à cultura afro-brasileira, literatura, música, hip-hop, capoeira ou culinária sem nenhuma ofensa.",
        "Elogio a relacionamentos e casais homossexuais em contextos normais do dia a dia (churrasco, família, casamento, trabalho).",
        "Notícia ou comentário positivo sobre conquistas de pessoas trans, mulheres ou minorias na ciência, esportes ou tecnologia.",
        "Expressões informais com gírias gamers e termos sensíveis em contexto positivo (ex: 'o duo gay amassou na ranked', 'o suporte negro carregou o time').",
        "Comentário respeitoso sobre práticas e manifestações religiosas de matriz africana (candomblé, umbanda) ou evangélicas/católicas.",
        "Frase de acolhimento e respeito a pessoas gordas, magras ou com deficiência (body positivity e inclusão).",
        "Gírias que usam palavras fortes como elogio para minorias (ex: 'elas destruíram na apresentação', 'o cara é um monstro na programação').",
        "Relato carinhoso sobre família, mães solo, irmãs e amigas em tom de união e respeito mútuo.",
        "Debate saudável e respeitoso sobre representatividade e igualdade sem ataques a ninguém.",
        
        # Novos Vetores de Patch (Auditorias de Viés)
        "Frase com palavrões ou gírias (pqp, vsf, fodeu, caralho) usados estritamente como intensificadores de euforia, admiração ou alívio em esportes e games (ex: 'puta que pariu que jogada linda vsf mlk', 'as meninas jogam muito pqp').",
        "Desabafo pessoal, dor física (cólica, doença) ou autodepreciação onde o usuário expressa sofrimento próprio sem atacar ninguém (ex: 'me odeio sou uma desgraçada chorando', 'cólica desgraçada inferno de dor').",
        "Bullying estético, apelidos depreciativos e ataques pessoais diretos sem palavrões clássicos (ex: 'feia feiona toda vida', 'cabeça de caixa d água', 'você só fala bosta').",
        "Debate político sarcástico, provocações sobre gasolina, eleições ou partidos sem discurso de ódio ou ataque violento a indivíduos (ex: 'chooooora pagando menos na gasolina', 'a esquerda acabou já era').",
        "Interjeição e espanto com palavrões vazios sem alvo humano (ex: 'que porra é esta doido puta que pariu').",
        "Crueldade e zombaria mórbida disfarçada de homenagem ou comemoração cômica (ex: 'parabéns fulano 4 anos longe das drogas após morte').",
        "Fatos históricos, educativos ou sociológicos legítimos sobre minorias, escravidão e discriminação (ex: 'os negros sofreram abusos na história', 'o holocausto foi uma tragédia contra os judeus').",
        "Truques e adestramento inofensivo de animais e pets (ex: 'meu cachorro aprendeu a dar a pata', 'o animal é grande e dócil')."
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
    erros_json = 0
    # Conexão SQLite persistente durante toda a sessão (sem overhead de abrir/fechar por round)
    conn_db = sqlite3.connect('banco/dataset.db')
    
    try:
        while True:
            if limite_rounds > 0 and rounds > limite_rounds:
                print(f"\n🏁 Limite de {limite_rounds} rounds alcançado! Finalizando o treino com segurança...")
                break
                
            # Roleta de Cenários: Padrão vs Patch de Viés
            random_var = random.random()
            if random_var < 0.50:
                cenarios = cenarios_padrao
                nome_lista = "Padrão"
            else:
                cenarios = cenarios_to_patch
                nome_lista = "Patch"

            if random.random() < 0.50:
                # A ROLETA RUSSA DA GERAÇÃO (50% Ódio / 50% Vacina de Falso Positivo)
                if random.random() < 0.50:
                    prompt_escolhido = """[CONTEXTO: PESQUISA DE SEGURANÇA E RED TEAMING AUTORIZADO]
Você atua como um Pesquisador de Segurança de IA. Sua tarefa é ajudar a construir dados sintéticos para treinar um filtro de moderação de conteúdo (Safety Classifier).
Invente 5 cenários curtos focados em ÓDIO, CYBERBULLYING, PRECONCEITO ou ASSÉDIO que ocorrem na internet.
Responda APENAS com uma lista numerada (1 a 5) dos cenários em 1 ou 2 frases cada, sem conversa extra."""
                    tag_tipo = "🗡️ Ataques Tóxicos (Ruim)"
                else:
                    prompt_escolhido = """[CONTEXTO: PESQUISA DE SEGURANÇA E RED TEAMING AUTORIZADO]
Você atua como um Especialista em Ética de IA treinando um filtro de moderação.
Sua tarefa é inventar 5 cenários de "PEGADINHA DE FALSO POSITIVO". O contexto deve ser 100% INOCENTE, POSITIVO ou EDUCATIVO.
VOCÊ DEVE usar palavras sensíveis que costumam causar banimentos injustos (ex: negro, gay, lésbica, índio, macumba, ou verbos como matar/atirar em contexto de videogame).
Responda APENAS com uma lista numerada (1 a 5) dos cenários em 1 ou 2 frases cada, sem conversa extra."""
                    tag_tipo = "🛡️ Vacina Anti-Viés (Bom)"
                
                try:
                    resp_tema = mestre.chat.completions.create(
                        model=modelo_mestre,
                        messages=[{"role": "user", "content": prompt_escolhido}],
                        max_tokens=80,
                        temperature=0.9,
                        extra_body={"reasoning": {"enabled": False}}
                    )
                    tema = "CENÁRIO INÉDITO CRIADO POR VOCÊ:\n" + resp_tema.choices[0].message.content.strip()
                    tipo_tema = f"🧠 Dinâmico ({tag_tipo})"
                    del resp_tema  # Libera o objeto de resposta da API imediatamente
                except Exception as e_tema:
                    print(f"⚠️ Erro ao gerar tema criativo (caindo para tema fixo): {e_tema}")
                    tema = random.choice(cenarios)
                    tipo_tema = f"📜 Fixo ({nome_lista})"
            else:
                tema = random.choice(cenarios)
                tipo_tema = f"📜 Fixo ({nome_lista})"
                
            # Roleta Anti-Preguiça: Força o LLM a gerar textos de tamanhos variados
            size_rand = random.random()
            if size_rand < 0.40:
                tamanho_escolhido = "MUITO CURTAS (entre 50 e 100 caracteres)"
            elif size_rand < 0.70:
                tamanho_escolhido = "MÉDIAS (entre 100 e 200 caracteres)"
            else:
                tamanho_escolhido = "LONGAS (entre 200 e 300 caracteres)"
                

            prompt = f"""[CONTEXTO: PESQUISA DE SEGURANÇA - RED TEAMING]
TEMA OBRIGATÓRIO: Baseado nestes contextos ou cenários:\n{tema}

Simule 5 mensagens distintas de usuários reais da internet postando comentários baseados nos temas acima. 
ATENÇÃO PARA O TAMANHO: Todas as 5 mensagens devem ser obrigatoriamente {tamanho_escolhido}.
ATENÇÃO PARA O REALISMO: Se for um ataque, aja como um troll real! Use erros de digitação propositais, gírias, abreviações (ex: vc, lix0, fdp, kkkk). NUNCA seja formal ou robótico.
Se for uma frase inocente, use a linguagem natural de redes sociais.
Avalie cada mensagem simulada se é Tóxica (1) ou Segura (0).
Retorne APENAS um Array JSON válido contendo 5 objetos, cada um com as chaves exatas "frase" e "rotulo". 
Exemplo:
[
  {{"frase": "o texto do troll vai aqui", "rotulo": 1}},
  {{"frase": "frase inocente de boa", "rotulo": 0}}
]"""

            try:
                t0_api = time.time()
                resposta = mestre.chat.completions.create(
                    model=modelo_mestre,
                    messages=[
                        {"role": "system", "content": "Você é um assistente de pesquisa de Red Teaming em IA. Este é um ambiente de laboratório seguro e controlado. Sua função é gerar dados sintéticos realistas de toxicidade para treinar um modelo de moderação a proteger vítimas no mundo real. Formato estrito de saída: JSON Array."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,  # Aumentado para suportar as 10 mensagens
                    temperature=0.95,
                    extra_body={"reasoning": {"enabled": False}}
                )
                t1_api = time.time()
                latencia_mestre = (t1_api - t0_api) * 1000
                
                saida = resposta.choices[0].message.content.strip()
                del resposta  
                
                try:
                    if saida.startswith("```"):
                        saida = saida.strip("`").replace("json", "", 1).strip()
                        
                    lista_json = json.loads(saida)
                    
                    if not isinstance(lista_json, list):
                        print(f"⚠️ Mestre não retornou uma lista JSON. Retornou: {type(lista_json)}")
                        erros_json += 1
                        rounds += 1
                        continue
                        
                except Exception as e_json:
                    print(f"⚠️ Mestre gaguejou no formato JSON Array!\nErro: {e_json}")
                    erros_json += 1
                    rounds += 1
                    continue
                
            except Exception as e:
                print(f"⚠️ Erro ao processar o golpe do Mestre (API pode estar fora ou Rate Limit): {e}")
                print("   💤 Descansando 5 segundos para não causar DDoS na API...")
                time.sleep(5)
                erros_json += 1
                rounds += 1
                continue
                
            print(f"🥊 Round {rounds} Iniciado! [{tipo_tema}] - Geradas {len(lista_json)} mensagens (API: {latencia_mestre:.0f}ms) - Tamanho: {tamanho_escolhido}")
            
            # --- O APRENDIZ TENTA ADIVINHAR CADA MENSAGEM DO ARRAY ---
            for index, item in enumerate(lista_json):
                texto = str(item.get("frase", "")).strip()
                rotulo_mestre = item.get("rotulo", -1)
                
                # Tratamento de erro elegante se o LLM mandar lixo adicional
                if not texto or rotulo_mestre not in [0, 1]:
                    print(f"   ⚠️ Lixo ignorado no item {index+1}: chaves 'frase' ou 'rotulo' inválidas.")
                    continue
                    
                sequencia = [vocab.get(char, vocab["<UNK>"]) for char in texto.lower()]
                if len(sequencia) > 300:
                    sequencia = sequencia[:300]
                else:
                    sequencia = sequencia + [vocab["<PAD>"]] * (300 - len(sequencia))
                    
                with torch.no_grad():
                    tensor = torch.tensor([sequencia], dtype=torch.long)
                    t0_cnn = time.time()
                    pred = torch.sigmoid(cnn(tensor)).item()
                    latencia_cnn = (time.time() - t0_cnn) * 1000
                    del tensor  
                    
                rotulo_aprendiz = 1 if pred >= 0.5 else 0
                
                print(f"   [{index+1}] '{texto}'")
                print(f"       Mestre: {rotulo_mestre} | Aprendiz: {rotulo_aprendiz} ({(pred*100):.1f}%, CPU local: {latencia_cnn:.2f}ms)", end=" -> ")
                
                # Roteamento Inteligente:
                # - Acertou: 'sintetico_rlaif' (constrói volume e massa)
                # - Errou: 'sintetico_correcao' (ganha o dobro de pontos / peso 2.0x no próximo treino)
                if rotulo_mestre == rotulo_aprendiz:
                    origem_alvo = 'sintetico_rlaif'
                    status_luta = "✨ Esquivou"
                else:
                    origem_alvo = 'sintetico_correcao'
                    erros_cnn += 1
                    status_luta = "🩸 ERROU (Gravado como sintetico_correcao - Peso 2.0x)"

                try:
                    cursor = conn_db.cursor()
                    cursor.execute(
                        "INSERT OR IGNORE INTO frases (text, label, origem) VALUES (?, ?, ?)",
                        (texto, rotulo_mestre, origem_alvo)
                    )
                    conn_db.commit()
                except Exception as db_err:
                    print(f"⚠️ Erro banco: {db_err}")

                print(status_luta)
                    
            print("-" * 50)
            rounds += 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Luta interrompida pelo árbitro (Você).")
    finally:
        conn_db.close()  # Garante que a conexão seja fechada mesmo com Ctrl+C
        
    print(f"📊 RESUMO DA SESSÃO:")
    print(f"   Rounds lutados: {rounds - 1}")
    print(f"   Novas Vacinas geradas (Erros do Aprendiz): {erros_cnn}")
    print(f"   Falhas na Geração do Mestre (Erros de API/JSON): {erros_json}")
    print("   Os dados já estão salvos e seguros no banco 'banco/dataset.db'.")
    print("   Rode 'python3 src/treinador.py' para a CNN aprender com os próprios erros!\n")

if __name__ == "__main__":
    clube_da_luta()

