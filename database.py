import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Credentials and configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# SQLAlchemy database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Base class for SQLAlchemy models
Base = declarative_base()

# Define tables
class Files(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    fileNameAlias = Column(String, nullable=False)
    filePath = Column(String, nullable=False)
    createdAt = Column(DateTime, nullable=False)


class UserFiles(Base):
    __tablename__ = "user_has_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fileId = Column(Integer, ForeignKey("files.id"), nullable=False)
    userId = Column(Integer, nullable=False)


class QueryLogs(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    queries = Column(Text, nullable=False)
    userId = Column(Integer, nullable=False)
    createdAt = Column(DateTime, nullable=False)

# Function to create the database if it doesn't exist
def create_database():
    try:
        # Connect to the default `postgres` database
        connection = psycopg2.connect(
            dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        connection.autocommit = True  # Enable autocommit for creating the database
        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()

        if not exists:
            # Create the database if it does not exist
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database {DB_NAME} created successfully.")
        else:
            print(f"Database {DB_NAME} already exists.")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error creating database: {e}")

# Function to create tables if they don't exist
def create_tables():
    try:
        # Connect to the specific database
        engine = create_engine(DATABASE_URL, echo=True)
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    # Step 1: Create the database if it doesn't exist
    create_database()

    # Step 2: Create tables in the database if they don't exist
    create_tables()