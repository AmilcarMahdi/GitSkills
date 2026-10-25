import sqlite3
import pandas as pd

SQLITE_DB_FILE = r"D:\PHD\TACS\Data\gitskills.sqlite.db"

# Definition of the 7 fine-grained micro-topics matching your evaluation framework
fine_grained_topics = {
    'Python Testing (PyTest)': "content LIKE '%testing%' AND content LIKE '%pytest%'",
    'JS Testing (Jest/Mocha)': "content LIKE '%testing%' AND (content LIKE '%jest%' OR content LIKE '%mocha%')",
    'Docker Infrastructure':   "content LIKE '%deployment%' AND content LIKE '%docker%'",
    'K8s Orchestration':       "content LIKE '%deployment%' AND content LIKE '%kubernetes%'",
    'SQL Migration (Alembic)': "content LIKE '%database_management%' AND content LIKE '%alembic%'",
    'NoSQL Cache (Redis)':     "content LIKE '%database_management%' AND content LIKE '%redis%'",
    'Static Security Linting': "content LIKE '%security%' AND content LIKE '%bandit%'"
}

print("📊 Interrogating full GitSkills Database for bucket distribution counts...\n")

try:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    
    print(f"{'Functional Engineering Topic Domain':<30} | {'Total Matching Files (All DB Scope)':<25}")
    print("-" * 65)
    
    total_found = 0
    for topic_name, sql_condition in fine_grained_topics.items():
        # Execute an ultra-fast SELECT COUNT query on the full production corpus
        query = f"SELECT COUNT(file_sha) as file_count FROM artifacts WHERE {sql_condition}"
        df_count = pd.read_sql_query(query, conn)
        
        count_value = df_chunk = df_count['file_count'].values[0]
        total_found += count_value
        
        print(f"{topic_name:<30} | {count_value:<25,}")
        
    print("-" * 65)
    print(f"{'Total Integrated SOTA Pool':<30} | {total_found:<25,}\n")
    conn.close()

except Exception as e:
    print(f"❌ Execution halted: {e}")