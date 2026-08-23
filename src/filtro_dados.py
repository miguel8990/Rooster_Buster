import pandas as pd
from datasets import load_dataset
import os

print("Baixando e carregando JAugusto97/told-br...")
told_br = load_dataset("JAugusto97/told-br", trust_remote_code=True)

print("Baixando e carregando Paul/hatecheck-portuguese...")
hatecheck = load_dataset("Paul/hatecheck-portuguese", trust_remote_code=True)

# O told-br geralmente tem colunas 'text' e rótulos
df_told = pd.DataFrame(told_br['train'])
# O hatecheck geralmente tem a coluna 'test_case'
df_hatecheck = pd.DataFrame(hatecheck['test'])

print(f"\nTamanho original TOLD-BR (train): {len(df_told)}")
print(f"Tamanho original HateCheck (test): {len(df_hatecheck)}")

# Filtrando textos com até 100 caracteres
# (Usamos str.strip() para garantir que não contaremos espaços em branco inúteis no final)
df_told_filtrado = df_told[df_told['text'].astype(str).str.strip().str.len() <= 100]
df_hatecheck_filtrado = df_hatecheck[df_hatecheck['test_case'].astype(str).str.strip().str.len() <= 100]

# Carrega, filtra e limpa os dados sintéticos também!
try:
    df_sintetico = pd.read_csv("dados/dados_sinteticos.csv")
    tamanho_antes = len(df_sintetico)
    df_sintetico_filtrado = df_sintetico[df_sintetico['text'].astype(str).str.strip().str.len() <= 100]
except FileNotFoundError:
    df_sintetico_filtrado = pd.DataFrame()
    tamanho_antes = 0

print(f"\n--- APÓS O FILTRO DE 100 CARACTERES ---")
print(f"TOLD-BR filtrado: {len(df_told_filtrado)} frases")
print(f"HateCheck filtrado: {len(df_hatecheck_filtrado)} frases")
if tamanho_antes > 0:
    print(f"Dados Sintéticos filtrados: {len(df_sintetico_filtrado)} frases (eram {tamanho_antes})")

# Salvando localmente para podermos analisar e etiquetar
os.makedirs("dados", exist_ok=True)
df_told_filtrado.to_csv("dados/told_br_curto.csv", index=False)
df_hatecheck_filtrado.to_csv("dados/hatecheck_curto.csv", index=False)
if tamanho_antes > 0:
    df_sintetico_filtrado.to_csv("dados/dados_sinteticos.csv", index=False)

print("\n✅ Filtro concluído! Arquivos salvos em RBooster/dados/")
