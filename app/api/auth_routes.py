from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.core.security import create_access_token
from app.models.auth import LoginRequest, RegisterRequest, TokenResponse
from app.services.user_store import authenticate_user, create_user

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> TokenResponse:
    try:
        user = create_user(request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    token = create_access_token(email=user["email"], role=user["role"])
    return TokenResponse(access_token=token, email=user["email"], role=user["role"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(email=user["email"], role=user["role"])
    return TokenResponse(access_token=token, email=user["email"], role=user["role"])
