"""受证据约束的回答生成能力。"""

from app.generation.answering.base import AnswerGenerator
from app.generation.answering.grounded import GroundedAnswerGenerator

__all__ = ["AnswerGenerator", "GroundedAnswerGenerator"]
