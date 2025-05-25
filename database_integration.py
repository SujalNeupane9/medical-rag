from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from pathlib import Path
import logging
from typing import List, Optional, Union
# Import the SQLAlchemy models
from database import Files, UserFiles, QueryLogs, Base, DATABASE_URL

class DatabaseManager:
    def __init__(self):
        """
        Initialize database connection and session
        """
        # Create engine and session
        self.engine = create_engine(DATABASE_URL, echo=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _check_existing_user_file(self, session, file_path: str, user_id: int) -> Optional[int]:
        """
        Check if a file with the same path already exists for the user
        
        Args:
            session: SQLAlchemy session
            file_path: Full path to the uploaded file
            user_id: ID of the user uploading the file
        
        Returns:
            The existing file ID if found, None otherwise
        """
        existing_file = (
            session.query(Files)
            .join(UserFiles, Files.id == UserFiles.fileId)
            .filter(
                Files.filePath == str(file_path),
                UserFiles.userId == user_id
            )
            .first()
        )
        return existing_file.id if existing_file else None

    def add_file_to_database(self, file_path: str, file_name_alias: str, user_id: int) -> Union[int, None]:
        """
        Add a file to the database and create a user-file association
        
        Args:
            file_path: Full path to the uploaded file
            file_name_alias: User-friendly name for the file
            user_id: ID of the user uploading the file
        
        Returns:
            The ID of the inserted file record, or None if file already exists
        """
        session = self.SessionLocal()
        try:
            # Check if file already exists for this user
            existing_file_id = self._check_existing_user_file(session, file_path, user_id)
            if existing_file_id:
                logging.warning(f"File {file_path} already exists for user {user_id}")
                return None

            # Create a new Files record
            new_file = Files(
                fileNameAlias=file_name_alias,
                filePath=str(file_path),
                createdAt=datetime.now()
            )
            session.add(new_file)
            session.flush()  # This will generate the ID without committing

            # Create a UserFiles association
            user_file = UserFiles(
                fileId=new_file.id,
                userId=user_id
            )
            session.add(user_file)
            
            session.commit()
            return new_file.id
        except Exception as e:
            session.rollback()
            logging.error(f"Error adding file to database: {e}")
            raise
        finally:
            session.close()

    def log_query(self, query: str, user_id: int):
        """
        Log user queries to the database
        
        Args:
            query: The user's query string
            user_id: ID of the user making the query
        """
        session = self.SessionLocal()
        try:
            query_log = QueryLogs(
                queries=query,
                userId=user_id,
                createdAt=datetime.now()
            )
            session.add(query_log)
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Error logging query: {e}")
        finally:
            session.close()

    def get_user_files(self, user_id: int) -> List[dict]:
        """
        Retrieve files uploaded by a specific user
        
        Args:
            user_id: ID of the user
        
        Returns:
            List of dictionaries containing file information
        """
        session = self.SessionLocal()
        try:
            # Join Files and UserFiles to get files for the specific user
            user_files = (
                session.query(Files)
                .join(UserFiles, Files.id == UserFiles.fileId)
                .filter(UserFiles.userId == user_id)
                .all()
            )
            
            # Convert to list of dictionaries
            return [
                {
                    "id": file.id,
                    "fileName": file.fileNameAlias,
                    "filePath": file.filePath,
                    "createdAt": file.createdAt
                } 
                for file in user_files
            ]
        except Exception as e:
            logging.error(f"Error retrieving user files: {e}")
            return []
        finally:
            session.close()

    def delete_file_from_database(self, file_id: int, user_id: int):
        """
        Delete a file record and its associated user file entry
        
        Args:
            file_id: ID of the file to delete
            user_id: ID of the user deleting the file
        """
        session = self.SessionLocal()
        try:
            # First, delete the UserFiles association
            session.query(UserFiles).filter(
                UserFiles.fileId == file_id,
                UserFiles.userId == user_id
            ).delete()

            # Then delete the Files record
            session.query(Files).filter(Files.id == file_id).delete()
            
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Error deleting file from database: {e}")
            raise
        finally:
            session.close()