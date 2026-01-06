import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException
from src.api.deps import get_current_user
from src.db.models import User
import uuid

@pytest.mark.asyncio
async def test_get_current_user_existing():
    # Setup mock DB
    mock_db = AsyncMock()
    user_id = uuid.uuid4()
    mock_user = User(id=user_id, email="test@example.com")

    # Mock execute/scalars/first result
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = mock_user
    mock_db.execute.return_value = mock_result

    # Call dependency
    user = await get_current_user(x_user_id=str(user_id), db=mock_db)

    assert user.id == user_id
    assert user.email == "test@example.com"
    # Ensure no create happened
    mock_db.add.assert_not_called()

@pytest.mark.asyncio
async def test_get_current_user_create_new():
    # Setup mock DB
    mock_db = AsyncMock()
    user_id = uuid.uuid4()

    # Mock execute/scalars/first result returning None
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    mock_db.execute.return_value = mock_result

    # Call dependency
    user = await get_current_user(x_user_id=str(user_id), db=mock_db)

    assert user.id == user_id
    assert user.tier == "free"
    # Ensure create happened
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_get_current_user_invalid_uuid():
    mock_db = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await get_current_user(x_user_id="invalid-uuid", db=mock_db)
    assert exc.value.status_code == 400
