# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

import schemas
from controllers.user import get_db
from db.update_db import update_database

update_database()
app = FastAPI()


@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
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
