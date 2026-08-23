import torch
import torch.nn as nn

class ModeradorCNN(nn.Module):
    """
    Arquitetura Convolucional 1D (CNN-1D) para leitura de caracteres.
    Ela lê 'janelas' de 3, 4 e 5 letras ao mesmo tempo para identificar 
    padrões visuais de palavrões, ofensa e burla de filtros.
    """
    def __init__(self, vocab_size, embedding_dim=64, num_filtros=128):
        super(ModeradorCNN, self).__init__()
        
        # 1. Dicionário Embutido (Embedding): Converte a ID da letra em um vetor matemático
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. Convoluções: As "lentes de aumento" que varrem a frase.
        # Adicionamos a conv2 graças à sua visão de detecção de ofensas de 2 letras!
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=4, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filtros, kernel_size=5, padding=2)
        
        # 3. Lóbulo Frontal de Decisão (O Cérebro Profundo) - Expandido para 1024 Neurônios!
        # Recebe (4 janelas convolucionais) * (Max + Avg) = 8 filtros!
        self.fc1 = nn.Linear(num_filtros * 8, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.fc5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # A 6ª camada que fará o julgamento final
        self.fc6 = nn.Linear(1024, 1)
        
        # O "Esquecimento Programado" (Anti-Decoréba)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x tem formato: [Lote, 300 caracteres]
        x_emb = self.embedding(x) 
        # Após embutir: [Lote, 300 caracteres, 32 Dimensões]
        
        # A Conv1d no PyTorch exige que a dimensão seja [Lote, Canais, Comprimento]
        x_emb = x_emb.permute(0, 2, 1)
        
        # Lente 0 (Duplas de letras)
        c2 = torch.relu(self.conv2(x_emb))
        c2_max = torch.max(c2, dim=2)[0]
        c2_avg = torch.mean(c2, dim=2)
        
        # Lente 1 (Trincas de letras)
        c3 = torch.relu(self.conv3(x_emb))
        c3_max = torch.max(c3, dim=2)[0] 
        c3_avg = torch.mean(c3, dim=2)
        
        # Lente 2 (Quartetos de letras)
        c4 = torch.relu(self.conv4(x_emb))
        c4_max = torch.max(c4, dim=2)[0] 
        c4_avg = torch.mean(c4, dim=2)
        
        # Lente 3 (Quintetos de letras)
        c5 = torch.relu(self.conv5(x_emb))
        c5_max = torch.max(c5, dim=2)[0] 
        c5_avg = torch.mean(c5, dim=2)
        
        # Junta Provas Criminais de Pico Máximo + O Tom Geral (Média) da Frase
        features = torch.cat((c2_max, c2_avg, c3_max, c3_avg, c4_max, c4_avg, c5_max, c5_avg), dim=1)
        
        # O Raciocínio (Passando pelos 6 andares do Lóbulo Frontal com Filtros de Ruído)
        x_fc = self.dropout(torch.relu(self.bn1(self.fc1(features))))
        x_fc = self.dropout(torch.relu(self.bn2(self.fc2(x_fc))))
        x_fc = self.dropout(torch.relu(self.bn3(self.fc3(x_fc))))
        x_fc = self.dropout(torch.relu(self.bn4(self.fc4(x_fc))))
        x_fc = self.dropout(torch.relu(self.bn5(self.fc5(x_fc))))
        
        # Retorna o "Logit" bruto (sem Sigmoid), pois o PyTorch 2.0 otimiza a matemática na BCEWithLogitsLoss
        saida = self.fc6(x_fc)
        
        return saida

def contar_parametros(modelo):
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total * 4) / (1024 * 1024)
    print("========================================")
    print(f"🧠 [ARQUITETURA CNN-1D] Parâmetros: {total}")
    print(f"📦 [PESO DO MODELO] Estimativa: {peso_mb:.4f} MB")
    print("========================================\n")
    return total
