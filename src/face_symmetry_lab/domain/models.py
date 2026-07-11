from enum import Enum
from typing import Optional
from pydantic import BaseModel

# C++の enum Status { OK, HOLD, ERROR }; に相当
# Enum: 決まった選択肢(定数)を定義可能
# ここで言うOK,HOLD,ERRORを指す
# Enumを継承
# OK,HOLD,ERRORは、それぞれオブジェクト、関数ではない。
class Status(Enum):
    OK = "OK"
    HOLD = "HOLD"
    ERROR = "ERROR"

class ReasonCode(Enum):
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    POSTURE = "POSTURE"
    DISTANCE = "DISTANCE"
    LOW_QUALITY = "LOW_QUALITY"
    
# それぞれの変数の型を定義
# BaseModelの継承を行っているので、初期化に必要なdef __init__()を書く必要がない
class AnalysisState(BaseModel):
    status: Status # statusという変数がStatus型を持っている
    # Optional[ReasonCode]: ReasonCodeの中身もしくはNoneが入る
    # =None: ディフォルトでNoneが入る
    # reason: 上記の型を持つ変数
    reason: Optional[ReasonCode] = None

class SymmetryScore(BaseModel):
    contour: float  # 輪郭のスコア
    eyes: float     # 目のスコア
    nose: float     # 鼻のスコア
    mouth: float    # 口のスコア
    overall: float  # 全体のスコア