import torch
import torch.nn as nn

class ModeradorCNN(nn.Module):
    """
    Arquitetura Convolucional 1D (CNN-1D).
    Ao invés de ler imagens, essa IA lê "janelas" de letras em um texto.
    Ela procura por padrões visuais de palavrões (ex: v,t,n,c) e tons de ofensa.
    """
    def __init__(self, vocab_size, embedding_dim=64, num_filtros=64):
        super(ModeradorCNN, self).__init__()
        
        # 1. CAMADA DE EMBUTIMENTO (Embedding)
        # O computador não lê a letra 'A'. Ele recebe o ID '10'. 
        # O Embedding é um Dicionário que transforma o ID '10' num vetor de 64 dimensões.
        # É aqui que a rede aprende que certas letras ou símbolos têm "significados" parecidos.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. LENTES CONVOLUCIONAIS (Filtros de Padrões)
        # Conv1d desliza sobre o texto como uma janela.
        # kernel_size=2: Lê de 2 em 2 letras (excelente para abreviações como "fd", "vs").
        # kernel_size=3: Lê de 3 em 3 letras (sílabas curtas).
        # Cada 'lente' tem 128 filtros, ou seja, ela caça 128 padrões tóxicos diferentes.
        # Trocamos o padding='same' (que derruba a velocidade na CPU) por paddings fixos simétricos (1 e 2).
        # Como usamos Global Pooling depois, o tamanho não precisa ser estritamente 300, 
        # e isso destrava a aceleração de hardware nativa (AVX2/MKL) do seu Ryzen!
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=5, padding=2)
        
        # 3. O CÉREBRO PROFUNDO (Lóbulo Frontal / Fully Connected Layers)
        # Após a CNN encontrar os "indícios" criminais nas letras, nós juntamos tudo.
        # 4 lentes * 2 tipos de análise (Max e Avg) = 8 pacotes de informação.
        # Multiplicado pelos 128 filtros = 1024 conexões de entrada.
        self.fc1 = nn.Linear(num_filtros * 8, 512)
        self.bn1 = nn.BatchNorm1d(512) # Normaliza os dados (impede que números explodam)
        
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        # CAMADA FINAL (A Decisão)
        # Transforma os 512 neurônios em apenas 1 único número: o Veredito (Tóxico ou Seguro).
        self.fc_final = nn.Linear(512, 1)
        
        # DROPOUT (Amnésia Programada)
        # Desliga 30% dos neurônios aleatoriamente durante o treino. 
        # Isso força a rede a não depender de neurônios viciados (decoréba) e a generalizar melhor.
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        O 'forward' é o caminho que a frase faz da entrada até a saída da IA.
        """
        # Entrada x: Lote de frases traduzidas em números [ex: 512 frases, 300 letras]
        
        # Transforma IDs simples nas matrizes profundas de 64 dimensões.
        x_emb = self.embedding(x) 
        
        # O PyTorch exige que os 'Canais' fiquem no meio. 
        # Permute inverte as dimensões de [512, 300, 64] para [512, 64, 300].
        x_emb = x_emb.permute(0, 2, 1)
        
        # Lente de 2 letras
        c2 = torch.relu(self.conv2(x_emb)) # ReLU = ignora números negativos
        c2_max = torch.max(c2, dim=2)[0]   # Pooling Máximo: Pega o sinal MAIS FORTE que essa lente achou na frase
        c2_avg = torch.mean(c2, dim=2)     # Pooling Médio: Tira uma média do tom geral da frase
        
        # Lente de 3 letras
        c3 = torch.relu(self.conv3(x_emb))
        c3_max = torch.max(c3, dim=2)[0] 
        c3_avg = torch.mean(c3, dim=2)
        
        # Lente de 4 letras
        c4 = torch.relu(self.conv4(x_emb))
        c4_max = torch.max(c4, dim=2)[0] 
        c4_avg = torch.mean(c4, dim=2)
        
        # Lente de 5 letras
        c5 = torch.relu(self.conv5(x_emb))
        c5_max = torch.max(c5, dim=2)[0] 
        c5_avg = torch.mean(c5, dim=2)
        
        # CONCATENAÇÃO: Junta as provas fortes (Max) e os contextos (Avg) de todas as lentes num só vetor.
        features = torch.cat((c2_max, c2_avg, c3_max, c3_avg, c4_max, c4_avg, c5_max, c5_avg), dim=1)
        
        # O PENSAMENTO PROFUNDO
        # A informação passa pelos 5 andares neurais, sofrendo filtragem (BatchNorm) e esquecimento (Dropout)
        x_fc = self.dropout(torch.relu(self.bn1(self.fc1(features))))
        x_fc = self.dropout(torch.relu(self.bn2(self.fc2(x_fc))))
        x_fc = self.dropout(torch.relu(self.bn3(self.fc3(x_fc))))
        x_fc = self.dropout(torch.relu(self.bn4(self.fc4(x_fc))))
        
        # SAÍDA BRUTA (Logit)
        # Não passamos a ativação Sigmoid aqui, apenas o número bruto. 
        # A função de Perda (BCEWithLogitsLoss) do PyTorch é mais rápida calculando Sigmoid por lá!
        saida = self.fc_final(x_fc)
        
        return saida

def contar_parametros(modelo):
    """
    Soma todos os neurônios e conexões que podem aprender algo (requires_grad).
    """
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total * 4) / (1024 * 1024)
    print("========================================")
    print(f"🧠 [ARQUITETURA CNN-1D] Parâmetros: {total}")
    print(f"📦 [PESO DO MODELO] Estimativa: {peso_mb:.4f} MB")
    print("========================================\n")
    return total
