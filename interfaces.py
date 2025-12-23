"""
Interfaces and Protocols for Trading Bot
این فایل interfaces را تعریف می‌کند تا coupling کاهش یابد
"""
from typing import Protocol, Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Import types (will be available after modularization)
try:
    from .types import NDSSignal, TradeInfo, AccountInfo
except ImportError:
    # Fallback for current single-file structure
    pass


# ============================================================================
# PROTOCOLS (Structural Subtyping)
# ============================================================================

class IMT5Manager(Protocol):
    """Interface for MT5 connection and data"""
    symbol: str
    account_info: Any
    
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_ohlcv(self, timeframe: int, count: int) -> Any: ...
    def get_current_price(self) -> tuple: ...
    def get_active_positions(self) -> List[Any]: ...
    def test_connection(self) -> bool: ...


class IRiskManager(Protocol):
    """Interface for Risk Management"""
    max_risk_percent: float
    max_lots: float
    
    def can_trade(self) -> tuple: ...
    def calculate_position_size(self, entry: float, stop_loss: float) -> float: ...
    def validate_signal(self, signal: Any) -> tuple: ...


class ITradeManager(Protocol):
    """Interface for Trade Operations"""
    def open_trade_safe(self, signal: Any) -> Optional[int]: ...
    def close_trade(self, trade: Any, reason: str) -> bool: ...
    def update_trailing_stop(self, trade: Any, new_sl: float) -> bool: ...


class IAnalyzer(Protocol):
    """Interface for Market Analysis"""
    def analyze(self) -> Optional[Any]: ...
    def optimized_analyze(self) -> Optional[Any]: ...


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def analyze(self) -> Optional[Any]:
        """Analyze market and generate signal"""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Any) -> bool:
        """Validate trading signal"""
        pass


class TradeExecutor(ABC):
    """Abstract base class for trade execution"""
    
    @abstractmethod
    def execute(self, signal: Any) -> Optional[int]:
        """Execute trade"""
        pass
    
    @abstractmethod
    def cancel(self, ticket: int) -> bool:
        """Cancel trade"""
        pass


# ============================================================================
# FACTORY INTERFACES
# ============================================================================

class IBotFactory(Protocol):
    """Interface for Bot Factory"""
    def create(self, bot_type: str, **kwargs) -> Any: ...


class IConfigLoader(Protocol):
    """Interface for Configuration Loading"""
    def load(self) -> Dict[str, Any]: ...
    def save(self, config: Dict[str, Any]) -> None: ...


# ============================================================================
# OBSERVER INTERFACE
# ============================================================================

class IObserver(Protocol):
    """Interface for Observer pattern"""
    def update(self, event: Any) -> None: ...


class ISubject(Protocol):
    """Interface for Subject in Observer pattern"""
    def attach(self, observer: IObserver) -> None: ...
    def detach(self, observer: IObserver) -> None: ...
    def notify(self, event: Any) -> None: ...


# ============================================================================
# REPOSITORY INTERFACE
# ============================================================================

class IMarketDataRepository(Protocol):
    """Interface for Market Data Access"""
    def get_ohlcv(self, timeframe: int, count: int) -> Any: ...
    def get_current_price(self) -> tuple: ...
    def get_tick(self) -> Any: ...


class ITradeRepository(Protocol):
    """Interface for Trade Data Access"""
    def get_active_positions(self) -> List[Any]: ...
    def get_position(self, ticket: int) -> Optional[Any]: ...
    def save_trade(self, trade: Any) -> None: ...


# ============================================================================
# EVENT INTERFACES
# ============================================================================

@dataclass
class BaseEvent:
    """Base class for all events"""
    timestamp: Any
    event_type: str


class IEventBus(Protocol):
    """Interface for Event Bus"""
    def subscribe(self, event_type: type, handler: callable) -> None: ...
    def publish(self, event: BaseEvent) -> None: ...
    def unsubscribe(self, event_type: type, handler: callable) -> None: ...

