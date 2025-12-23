"""
Custom Exceptions for Trading Bot
تعریف exception های اختصاصی برای مدیریت بهتر خطاها
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass


class MT5ConnectionError(TradingBotError):
    """MT5 connection related errors"""
    pass


class MT5DataError(TradingBotError):
    """MT5 data retrieval errors"""
    pass


class TradeExecutionError(TradingBotError):
    """Trade execution failed"""
    def __init__(self, message: str, retcode: int = None):
        super().__init__(message)
        self.retcode = retcode


class RiskManagementError(TradingBotError):
    """Risk management violation"""
    pass


class SignalValidationError(TradingBotError):
    """Signal validation failed"""
    pass


class ConfigurationError(TradingBotError):
    """Configuration related errors"""
    pass


class StateError(TradingBotError):
    """State management errors"""
    pass


class AnalysisError(TradingBotError):
    """Market analysis errors"""
    pass

