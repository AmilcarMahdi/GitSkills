import sqlite3

conn = sqlite3.connect("D:\\PHD\\TACS\\Data\\gitskills.sqlite.db")
cursor = conn.cursor()

try:
    cursor.execute("PRAGMA table_info(artifacts);")
    columns = cursor.fetchall()
    print("--- Colonnes dans la table 'artifacts' ---")
    for col in columns:
        print(f"Nom: {col[1]} | Type: {col[2]}")
except Exception as e:
    print(f"Erreur lors de la lecture de la table: {e}")

conn.close()
