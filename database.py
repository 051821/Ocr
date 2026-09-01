import config
import psycopg2.pool


db_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=config.DB_POOL_MIN,
    maxconn=config.DB_POOL_MAX,
    dsn=config.DATABASE_URL
)