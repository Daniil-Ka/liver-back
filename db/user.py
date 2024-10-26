import hashlib
import os
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import select

from db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)

    def verify_password(self, plain_password: str) -> bool:
        """Проверка пароля пользователя"""
        salt, stored_hash = self.password_hash.split('$')
        return stored_hash == hashlib.sha256(salt.encode() + plain_password.encode()).hexdigest()

    @staticmethod
    def hash_password(password: str) -> str:
        """Хэширование пароля с солью"""
        salt = os.urandom(16).hex()  # Генерация случайной соли
        hashed_password = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
        return f"{salt}${hashed_password}"