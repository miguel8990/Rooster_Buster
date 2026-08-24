# 🛡️ RBooster (Rooster Buster) - Filtro de Moderação de IA

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

O **RBooster** é um sistema de moderação de texto baseado em Inteligência Artificial, construído do zero com uma arquitetura **CNN-1D**. Ele foi projetado para ler textos, identificar padrões visuais de linguagem tóxica (ódio, racismo, xenofobia, assédio) e emitir um Veredito de Segurança em milissegundos.

O grande diferencial deste projeto é a sua **Otimização Extrema para CPU**. Ele foi forjado para treinar bases de dados de nível datacenter (4.3+ Milhões de registros) diretamente em processadores domésticos multi-core (como a linha AMD Ryzen), superando gargalos de memória RAM, threads e limites arquiteturais sem a necessidade de uma GPU dedicada.

---

## 🧠 Arquitetura do Modelo (CNN-1D)
Ao invés de ler imagens, o RBooster "lê" janelas de letras no texto, caçando anomalias.
A rede neural enxuta possui cerca de **2.003.521 de parâmetros**, garantindo altíssima inteligência sem sobrecarregar a largura de banda da memória DDR4.

1. **Camada de Embutimento (Embedding):** Transforma caracteres em vetores de 32 dimensões.
2. **Lentes Convolucionais (As Lupas):** 4 detectores simultâneos buscando padrões de 2, 3, 4 e 5 letras (ex: identificando abreviações maliciosas como `fdp` ou espaçamentos disfarçados).
3. **Global Max/Avg Pooling (O Esmagamento):** Extrai apenas o pico máximo de toxicidade encontrado na frase, tornando o modelo imune ao tamanho do texto (agnóstico a sequência).
4. **Comitê de Votação (Lóbulo Frontal):** 4 camadas densas (512 neurônios cada) que avaliam as evidências das convoluções e emitem o veredito (0.0 a 1.0).

---

## ⚡ Engenharia e Otimização Extrema (CPU)
Este projeto contém modificações de baixo nível para extrair o máximo de processadores **AMD Ryzen (Zen 3/Zen 4)**:

* **Blindagem contra Copy-on-Write (COW):** O DataLoader foi reescrito para utilizar arrays blocados em C (`numpy.int16`). Isso impede que o Linux duplique os 2.6GB de RAM ao usar múltiplos *Workers*, eliminando vazamentos de memória (Memory Leaks).
* **Fixação de Threads Físicas:** Restrição via `OMP_NUM_THREADS` para evitar que o Hyperthreading gere contenção de Cache L3 e destrua a velocidade do barramento interno.
* **Destravamento do Motor AVX2:** Remoção de tensores *BFloat16* (que forçavam emulação via software em CPUs sem AVX-512) e remoção de paddings assimétricos (`padding='same'`), destravando o processamento nativo nas FPUs em precisão `Float32`.
* **Adequação de L3 Cache:** Configuração de `batch_size=512` e `embedding_dim=32` para que os tensores intermediários caibam nos 16MB de Cache L3 do processador, mitigando o gargalo da memória RAM DDR4.

---

## 📂 Estrutura do Projeto

```text
/workspace
├── banco/               # Banco de dados (SQLite) com milhões de frases
├── pesos/               # Onde os "Cérebros" treinados (.pth) são salvos
├── tutorias/            # Documentação técnica e artigos de engenharia do modelo
├── src/                 
│   ├── modelo.py        # Arquitetura Matemática (A CNN-1D)
│   ├── treinador.py     # Motor de Treinamento e Otimização
│   ├── interface.py     # Chat Interativo e Benchmark de Validação
│   ├── rlhf_humano.py   # Feedback Humano (Ensine o modelo quando ele errar)
│   └── rlaif_clube_da_luta.py # IA contra IA gerando dados sintéticos
└── vocabulario.json     # Dicionário mapeando letras para números
```

---

## 🚀 Como Usar

### 1. Treinamento
Para iniciar o treinamento da rede neural. O script varre o banco de dados, aloca os 4.3 milhões de textos na memória de forma otimizada e inicia as épocas.
```bash
python3 src/treinador.py
```

### 2. Interface (Testes e Uso)
Carrega os pesos treinados e permite testar frases no modo interativo ou rodar um benchmark rápido de 50 frases para medir Falsos Positivos e Falsos Negativos.
```bash
python3 src/interface.py
```

### 3. Loop de Feedback Humano (RLHF)
A IA baniu alguém injustamente? Ou deixou um palavrão passar? Rode este script para corrigir o erro. A sua correção entrará no banco de dados com **Peso VIP (8x)**, forçando o modelo a respeitar a sua decisão no próximo treino.
```bash
python3 src/rlhf_humano.py
```

---

