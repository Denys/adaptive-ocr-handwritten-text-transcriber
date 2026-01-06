from typing import Generator
from fastapi import Depends, HTTPException, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.db.session import get_db
from src.db.models import User
import uuid

async def get_current_user(
    x_user_id: str = Header(..., description="User ID for MVP/Testing"),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current user.
    For MVP Week 2, we simulate authentication by trusting the X-User-ID header.
    In production, this would verify a JWT token.
    """
    try:
        user_uuid = uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid User ID format"
        )

    # For MVP: If user doesn't exist, auto-create (to simplify testing flow)
    # This acts as a lazy registration
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalars().first()

    if not user:
        # Create new user for testing
        user = User(
            id=user_uuid,
            email=f"user_{x_user_id[:8]}@example.com", # Dummy email
            tier="free"
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user
