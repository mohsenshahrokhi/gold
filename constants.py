"""
Constants and Magic Numbers
تمام مقادیر ثابت در یک مکان مرکزی
"""
from enum import IntEnum

# ============================================================================
# TRADING CONSTANTS
# ============================================================================

# Risk Management
DEFAULT_MAX_RISK_PERCENT = 0.5
DEFAULT_MAX_LOTS = 0.3
DEFAULT_MIN_BALANCE = 500.0
DEFAULT_MAX_DAILY_LOSS = 0.02
DEFAULT_MAX_DAILY_TRADES = 10

# Trading
DEFAULT_MAGIC_NUMBER = 123456
DEFAULT_PIP_MARGIN = 5
DEFAULT_MIN_VOLUME = 0.01
DEFAULT_MAX_VOLUME = 0.5
DEFAULT_SPREAD_MARGIN = 5.0
DEFAULT_COMMISSION = 0.0

# Scalping
DEFAULT_MAX_SCALP_TRADES_PER_HOUR = 10
DEFAULT_SCALP_TARGET_PIPS = 10
DEFAULT_SCALP_MAX_RISK_PIPS = 5
DEFAULT_MIN_SCALP_CONFIDENCE = 0.7
DEFAULT_SCALP_FIXED_VOLUME = 0.1

# NDS Parameters
DEFAULT_ALPHA_CORRECTION = 0.86
DEFAULT_ALPHA_PRESSURE = 0.2
DEFAULT_BETA_DISPLACEMENT = 0.3

# Timeframes (MT5)
TIMEFRAME_M1 = 1
TIMEFRAME_M3 = 3
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 60

# ============================================================================
# SLEEP INTERVALS
# ============================================================================

SLEEP_CONNECTION_RETRY = 1
SLEEP_AFTER_ORDER = 2
SLEEP_POSITION_CHECK = 1
SLEEP_SCALP_LOOP = 0.5
SLEEP_MAIN_LOOP = 1
SLEEP_ERROR_RECOVERY = 5

# ============================================================================
# CACHE SETTINGS
# ============================================================================

CACHE_DEFAULT_TTL = 60
CACHE_MAX_SIZE = 1000

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

MIN_RISK_REWARD_RATIO = 1.5
MIN_SIGNAL_CONFIDENCE = 0.6
MIN_SCALP_CONFIDENCE = 0.7

# ============================================================================
# ENUMS
# ============================================================================

class BotType(IntEnum):
    """Bot type enumeration"""
    ORIGINAL = 1
    ENHANCED = 2
    OPTIMIZED = 3
    SCALPING = 4
    PROFESSIONAL_SCALPING = 5

