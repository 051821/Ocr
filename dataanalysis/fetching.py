import config
import psycopg2 
import psycopg2.extras 
import psycopg2.pool

db = config.DATABASE_URL 

curr = psycopg2.connect(db)

def fetch_batch():
    cursor = curr.cursor()
    cursor.execute("SELECT * FROM document_extraction limit 2")

