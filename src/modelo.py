import torch
import torch.nn as nn

class ModeradorCNN(nn.Module):
    """
    Arquitetura Convolucional 1D Multifocal (CNN-1D).
    Possui 9 Lentes Estratégicas de 2 até 24 caracteres:
    - [2, 3, 4, 5]: Gírias ultracurtas, radicais e palavras isoladas (ex: vsf, l1x0, fdp).
    - [7, 10]: Palavras longas e expressões curtas (ex: cala a boca, gente boa).
    - [15, 18, 24]: Expressões complexas e orações inteiras (ex: vai se fuder seu lixo, excelente dev).
    """
    def __init__(self, vocab_size, embedding_dim=256, num_filtros=512, kernel_sizes=(2, 3, 4, 5, 7, 10, 15, 18, 24)):
        super(ModeradorCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        
        # 1. CAMADA DE EMBUTIMENTO (Embedding)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. LENTES CONVOLUCIONAIS MULTIFOCAIS (9 Lentes)
        # Cada lente possui 512 filtros especializados para o seu tamanho de janela.
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        
        # 3. O CÉREBRO PROFUNDO (Lóbulo Frontal / Fully Connected Layer)
        # 9 lentes * 2 tipos de análise (Max e Avg) = 18 pacotes de informação por filtro.
        # 18 * 128 filtros = 2.304 conexões de entrada.
        num_features_in = len(kernel_sizes) * 2 * num_filtros
        self.fc1 = nn.Linear(num_features_in, 1024)
        self.bn1 = nn.BatchNorm1d(1024) # Normaliza os dados (impede que números explodam)
        
        # CAMADA FINAL (A Decisão)
        # Transforma os 1024 neurônios em apenas 1 único número: o Veredito (Tóxico ou Seguro).
        self.fc_final = nn.Linear(1024, 1)
        
        # DROPOUT (Amnésia Programada)
        # Desliga 30% dos neurônios aleatoriamente durante o treino contra o decoréba.
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        O caminho que a frase faz da entrada até a saída da IA.
        """
        # Transforma IDs simples em matrizes de 64 dimensões.
        x_emb = self.embedding(x) 
        
        # Permute inverte as dimensões de [batch, 300, 64] para [batch, 64, 300].
        x_emb = x_emb.permute(0, 2, 1)
        
        # Passa o texto simultaneamente por todas as 9 Lentes Multifocais
        pooled_features = []
        for conv in self.convs:
            c = torch.relu(conv(x_emb))
            c_max = torch.max(c, dim=2)[0] # Pega o sinal MAIS FORTE que essa lente achou na frase
            c_avg = torch.mean(c, dim=2)   # Tira uma média do tom geral da frase
            pooled_features.append(c_max)
            pooled_features.append(c_avg)
            
        # CONCATENAÇÃO: Junta todas as 18 provas fortes (Max) e contextos (Avg) em um só vetor.
        features = torch.cat(pooled_features, dim=1)
        
        # PENSAMENTO E DECISÃO
        x_fc = self.dropout(torch.relu(self.bn1(self.fc1(features))))
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
