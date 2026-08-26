import sqlite3
import os

# Função que exibe o status do banco de dados (RBooster)
def exibir_status_banco():
    db_path = 'banco/dataset.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Banco de dados não encontrado em {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        print("="*40)
        print(" 📊 STATUS DO BANCO DE DADOS (RBooster)")
        print("="*40)
        
        # 1. Total Geral
        cur.execute('SELECT COUNT(*) FROM frases')
        total = cur.fetchone()[0]
        print(f"\n📈 TOTAL DE FRASES: {total:,}".replace(',', '.'))
        
        if total == 0:
            print("\nO banco de dados está vazio no momento.")
            return
        
        # 2. Divisão por Origem
        print("\n📂 DIVISÃO POR DATASET (Origem):")
        cur.execute('SELECT origem, COUNT(*) FROM frases GROUP BY origem ORDER BY COUNT(*) DESC')
        for origem, count in cur.fetchall():
            print(f"   - {origem}: {count:,}".replace(',', '.'))
            
        # 3. Divisão por Rótulo (Seguro vs Tóxico)
        print("\n⚖️ BALANÇO DE CLASSES:")
        cur.execute('''
            SELECT 
                CASE WHEN CAST(label AS REAL) >= 0.5 THEN 'Tóxico (1)' ELSE 'Seguro (0)' END as classe,
                COUNT(*) 
            FROM frases 
            WHERE CAST(label AS REAL) IS NOT NULL
            GROUP BY classe
        ''')
        
        for classe, count in cur.fetchall():
            icone = "🔴" if "Tóxico" in classe else "🟢"
            porcentagem = (count / total) * 100
            print(f"   {icone} {classe}: {count:,}".replace(',', '.') + f" ({porcentagem:.1f}%)")
            
        print("\n" + "="*40 + "\n")
        
    except sqlite3.Error as e:
        print(f"❌ Erro ao ler o banco: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    exibir_status_banco()
