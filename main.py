# main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from starlette.middleware.cors import CORSMiddleware
#from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import schemas
from controllers import auth
from db.database import get_db
from db.update_db import update_database
from db.user import User


# обновление таблиц БД
update_database()

app = FastAPI()

# Настройки CORS
origins = [
    "http://localhost:5174",  # URL фронтенд-приложения
]
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(email=user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    return auth.create_user(db=db, user=user)


@app.post("/login", response_model=schemas.Token)
def login(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = auth.authenticate_user(db, user.email, user.password)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Здесь можно добавить генерацию JWT токена
    return {
        "access_token": "dummy_token",  # Здесь должен быть настоящий токен
        "token_type": "bearer",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
