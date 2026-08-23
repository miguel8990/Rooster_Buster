import torch
import torch.nn as nn

class ModeradorCNN(nn.Module):
    """
    Arquitetura Convolucional 1D (CNN-1D) para leitura de caracteres.
    Ela lê 'janelas' de 3, 4 e 5 letras ao mesmo tempo para identificar 
    padrões visuais de palavrões, ofensa e burla de filtros.
    """
    def __init__(self, vocab_size, embedding_dim=32, num_filtros=64):
        super(ModeradorCNN, self).__init__()
        
        # 1. Dicionário Embutido (Embedding): Converte a ID da letra em um vetor matemático
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. Convoluções: As "lentes de aumento" que varrem a frase.
        # Adicionamos a conv2 graças à sua visão de detecção de ofensas de 2 letras!
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=4, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=5, padding=2)
        
        # 3. Funil de Decisão Final
        # Juntamos as descobertas das 4 lentes agora (64 filtros * 4 = 256 sinais)
        self.fc = nn.Linear(num_filtros * 4, 1)
        
        # A Sigmoid garante que a resposta final fique perfeitamente espremida entre 0.0 (Anjo) e 1.0 (Banido)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x tem formato: [Lote, 100 caracteres]
        x_emb = self.embedding(x) 
        # Após embutir: [Lote, 100 caracteres, 32 Dimensões]
        
        # A Conv1d no PyTorch exige que a dimensão seja [Lote, Canais, Comprimento]
        x_emb = x_emb.permute(0, 2, 1)
        
        # Lente 0 (Duplas de letras) - Captura ofensas minúsculas!
        c2 = torch.relu(self.conv2(x_emb))
        c2 = torch.max(c2, dim=2)[0]
        
        # Lente 1 (Trincas de letras) + ReLU + Pega a maior ativação
        c3 = torch.relu(self.conv3(x_emb))
        c3 = torch.max(c3, dim=2)[0] 
        
        # Lente 2 (Quartetos de letras)
        c4 = torch.relu(self.conv4(x_emb))
        c4 = torch.max(c4, dim=2)[0] 
        
        # Lente 3 (Quintetos de letras)
        c5 = torch.relu(self.conv5(x_emb))
        c5 = torch.max(c5, dim=2)[0] 
        
        # Junta todas as "provas" criminais que a rede achou
        features = torch.cat((c2, c3, c4, c5), dim=1)
        
        # Dá a sentença final
        saida = self.fc(features)
        return self.sigmoid(saida)

def contar_parametros(modelo):
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total * 4) / (1024 * 1024)
    print("========================================")
    print(f"🧠 [ARQUITETURA CNN-1D] Parâmetros: {total}")
    print(f"📦 [PESO DO MODELO] Estimativa: {peso_mb:.4f} MB")
    print("========================================\n")
    return total
