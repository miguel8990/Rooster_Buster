from llama_cpp import Llama
import os
import sys

def iniciar_gerador():
    print("==================================================")
    print(" 🏭 FÁBRICA DE DADOS SINTÉTICOS OFFLINE (Llama.cpp)")
    print("==================================================")
    print("Carregando o colossal Ministral 3 14B da Memória Física (Offline)...")
    
    # Como você vai organizar tudo certinho, ele vai buscar direto na pasta pesos!
    # Lembre-se de verificar se o nome final do arquivo é exatamente esse abaixo:
    caminho_do_arquivo_gguf = "pesos/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf" 
    
    if not os.path.exists(caminho_do_arquivo_gguf):
        print(f"\n❌ ERRO: O arquivo GGUF não foi encontrado em: {caminho_do_arquivo_gguf}")
        print("Edite o arquivo 'src/gerador_sintetico_local.py' e coloque o caminho exato de onde você salvou o download!")
        sys.exit(1)
        
    try:
        print("\nDespejando os 14 Bilhões de parâmetros na Memória RAM...")
        llm = Llama(
            model_path=caminho_do_arquivo_gguf,
            n_ctx=2048,
            n_threads=16, # Usa todos os seus 16 núcleos!
            use_mlock=True, # Bloqueia o modelo na RAM (Impede o Linux de jogar pro Swap)
            verbose=False
        )
    except Exception as e:
        print("Erro ao carregar o modelo. Verifique sua conexão com a internet para o download inicial.")
        print(e)
        sys.exit(1)
        
    prompt = """Gere EXATAMENTE 5 frases curtas (menos de 80 caracteres) em português brasileiro realista de redes sociais.
Intercale entre frases normais/amigáveis (rótulo 0) e xingamentos/ódio pesado com erros propositais/abreviações como vtnc, fdp, lix0 (rótulo 1).
Exemplo estrito:
tu é muito gente boa mano,0
cala a boca seu lixo humano,1
Formato: FRASE,ROTULO
Responda APENAS com as 5 frases no formato CSV. NENHUMA PALAVRA A MAIS. NÃO USE ASPAS."""

    caminho_arquivo = "dados/dados_sinteticos_offline.csv"
    
    # Se o arquivo não existir, cria com cabeçalho
    if not os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write("text,label\n")
            
    print("⚙️  Gerando dados em loop infinito. Aperte Ctrl+C quando achar que já tem o suficiente!\n")
    
    with open(caminho_arquivo, "a", encoding="utf-8") as f:
        lote = 1
        try:
            while True:
                print(f"[{lote}] Fabricando 5 frases novas...")
                resposta = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Você é uma máquina conversora que retorna apenas CSV bruto sem markdown e sem conversa."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.6 # Reduzido para evitar alucinações absurdas e loops
                )
                
                texto_gerado = resposta['choices'][0]['message']['content'].strip()
                linhas = texto_gerado.split('\n')
                
                frases_salvas = 0
                for linha in linhas:
                    linha = linha.strip()
                    # Regra de segurança para não salvar lixo gerado
                    if ',' in linha and (linha.endswith(',0') or linha.endswith(',1')):
                        # Separa a frase do rótulo para garantir as aspas de segurança (aquele erro que tivemos antes!)
                        idx_virgula = linha.rfind(',')
                        texto = linha[:idx_virgula].strip('\"').replace('"', '')
                        rotulo = linha[idx_virgula+1:].strip()
                        
                        f.write(f'"{texto}",{rotulo}\n')
                        frases_salvas += 1
                
                f.flush()
                print(f"   -> {frases_salvas} frases injetadas no CSV!")
                lote += 1
                
        except KeyboardInterrupt:
            print(f"\n🛑 Geração interrompida com segurança! Seus dados estão em: {caminho_arquivo}")
            print("Para incluí-los no treino, basta adicionar a leitura desse arquivo no treinador.py!")

if __name__ == "__main__":
    iniciar_gerador()
