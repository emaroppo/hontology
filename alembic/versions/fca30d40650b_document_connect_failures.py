"""document connect_failures

Revision ID: fca30d40650b
Revises: b0f801e569c8
Create Date: 2026-10-06 18:56:08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'fca30d40650b'
down_revision: str | None = 'b0f801e569c8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('connect_failures', sa.Integer(), server_default='0', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('documents', 'connect_failures')
