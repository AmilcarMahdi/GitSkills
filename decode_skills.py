import sqlite3

# Liste des 5 hashes obtenus dans le classement MarsRank
TOP_HASHES = [
    "a5a39371307e4872b4b3ce0408e05a38bee2876e",
    "5a6a964f456af254dbdffedca13e0584efd0fa7a",
    "68d2935f9cea78dd22caa555b7fa152a14eb36fa",
    "bed45bdc3c8385195f100dfb954c0208f2e4a0ca",
    "625a8eb495ec14b5c4a5d408d8eecd02d37ea4af"
]

DB_PATH = "D:\\PHD\\TACS\\Data\\gitskills.sqlite.db"

def decode_skills():
    print(f"Connecting to database to decode Top 5 Skills...\n")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for rank, file_sha in enumerate(TOP_HASHES, 1):
        # Requête SQL pour extraire l'identité de l'artefact
        query = """
            SELECT repo_full_name, path, filename, content 
            FROM artifacts 
            WHERE file_sha = ?
        """
        cursor.execute(query, (file_sha,))
        row = cursor.fetchone()
        
        print(f"==================================================")
        print(f"🏆 RANK {rank} | Hash: {file_sha}")
        print(f"==================================================")
        
        if row:
            repo, path, filename, content = row
            print(f"📁 Repository : {repo}")
            print(f"📄 File       : {filename}")
            print(f"📍 Full Path  : {path}")
            print(f"\n📝 Content Preview (First 15 lines) :")
            print("-" * 40)
            
            # Affichage des 15 premières lignes du code/frontmatter
            if content:
                lines = str(content).splitlines()[:15]
                for line in lines:
                    print(f"  {line}")
                if len(str(content).splitlines()) > 15:
                    print("  [...] (truncated)")
            else:
                print("  [Empty Content]")
        else:
            print(f"❌ [Warning] No metadata found in 'artifacts' table for this hash.")
            
        print("-" * 40 + "\n")
        
    conn.close()

if __name__ == "__main__":
    decode_skills()
