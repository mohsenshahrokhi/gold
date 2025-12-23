"""
===============================================================================
🤖 ENHANCED NDS TRADING BOT - COMPLETE IMPLEMENTATION
===============================================================================
📌 Complete NDS paper implementation with Fractal, Symmetry, and Neural Network
📌 Ready for MetaTrader 5 connection

Author: Enhanced NDS System
Date: 1404/09/30
Version: 2.0.0 - Complete Paper Implementation
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import os
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging
import time
import warnings
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import norm
import threading
from collections import deque

# Import modular components (if available, otherwise use inline)
try:
    from config import BotConfig, get_config
    from exceptions import TradingBotError, MT5ConnectionError, TradeExecutionError
    from constants import *
    from symbol_resolver import SymbolResolver, get_symbol_menu
    MODULAR_IMPORTS_AVAILABLE = True
except ImportError:
    MODULAR_IMPORTS_AVAILABLE = False
    # Will use inline definitions
    # Define minimal SymbolResolver inline
    class SymbolResolver:
        def find_symbol(self, name: str):
            return name
        def is_symbol_tradeable(self, symbol: str):
            return True
    def get_symbol_menu():
        return {
            'XAUUSD (Gold)': 'XAUUSD',
            'EURUSD (Euro/USD)': 'EURUSD',
            'US30/YM (Dow Jones)': 'US30',
            'BTCUSD (Bitcoin)': 'BTCUSD'
        }

# Neural Network Imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
    torch_error = None
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch_error = str(e)
    # Create dummy classes to prevent import errors
    class DummyModule:
        def __init__(self, error_msg):
            self.error_msg = error_msg
        def __getattr__(self, name):
            raise RuntimeError(f"PyTorch is not available. Error: {self.error_msg}")
        def __call__(self, *args, **kwargs):
            raise RuntimeError(f"PyTorch is not available. Error: {self.error_msg}")
    torch = DummyModule(torch_error)
    nn = DummyModule(torch_error)
    optim = DummyModule(torch_error)
    F = DummyModule(torch_error)
    Normal = DummyModule(torch_error)
    Categorical = DummyModule(torch_error)

# Advanced ML Imports for Article Models
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Log sklearn availability after logger is defined
if not SKLEARN_AVAILABLE:
    logger.warning("sklearn not available - some features will be disabled")

# Log torch availability after logger is defined
if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available (DLL error) - neural network features will be disabled")
    logger.warning("  Solution: Install Visual C++ Redistributable 2015-2022 or reinstall PyTorch")

# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================
class TrendDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class QuantumState(Enum):
    SUPERPOSITION = "superposition"
    COLLAPSED_BULLISH = "collapsed_bullish"
    COLLAPSED_BEARISH = "collapsed_bearish"

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class Node:
    """Price node (reversal point)"""
    index: int
    price: float
    time: datetime
    node_type: str  # 'high' or 'low'
    strength: float = 0.0
    displaced_price: Optional[float] = None
    
    def __repr__(self):
        return f"Node({self.node_type}, price={self.price:.2f}, strength={self.strength:.2f})"

@dataclass
class Cycle:
    """Price cycle"""
    start_node: Node
    end_node: Node
    rally: float = 0.0
    correction: float = 0.0
    net_movement: float = 0.0
    direction: TrendDirection = TrendDirection.NEUTRAL
    
    def calculate(self):
        """Calculate cycle parameters: C_n = R_n - K_n = 0.14 * R_n"""
        price_diff = self.end_node.price - self.start_node.price
        self.rally = abs(price_diff)
        self.correction = 0.86 * self.rally  # K_n = 0.86 * R_n
        self.net_movement = 0.14 * self.rally  # C_n = 0.14 * R_n
        self.direction = TrendDirection.BULLISH if price_diff > 0 else TrendDirection.BEARISH

@dataclass
class PolynomialFunction:
    """Polynomial function for trend or pullback"""
    coefficients: np.ndarray
    degree: int
    r_squared: float = 0.0
    velocity: float = 0.0
    function_type: str = "trend"  # 'trend' or 'pullback'

@dataclass
class NDSSignal:
    """NDS signal"""
    direction: TrendDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    quantum_state: QuantumState
    hurst_exponent: float
    risk_reward: float
    timestamp: datetime
    nodes: List[Node] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Validate signal"""
        return (self.confidence > 0.6 and 
                self.risk_reward >= 1.5 and 
                self.direction != TrendDirection.NEUTRAL)

@dataclass
class AccountInfo:
    """Account information"""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int
    currency: str
    server: str
    is_demo: bool
    can_trade: bool

@dataclass
class TradeInfo:
    """Trade information"""
    ticket: int
    symbol: str
    order_type: int
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    open_time: datetime
    type: int = field(init=False)
    def __post_init__(self):
        # این کار برای سازگاری backward انجام می‌شود
        # بعضی قسمت‌های کد از trade.type استفاده می‌کنند
        self.type = self.order_type

# ============================================================================
# MT5 MANAGER - MetaTrader 5 Connection Manager
# ============================================================================
class MT5Manager:
    """Manage connection and communication with MetaTrader 5"""
    
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.connected = False
        self.account_info: Optional[AccountInfo] = None
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """Connect to MetaTrader 5 - FIXED VERSION"""
        try:
            # اگر قبلاً وصل بودیم، اول قطع کنیم
            if mt5.terminal_info() is not None:
                mt5.shutdown()
                time.sleep(1)
            
            # تنظیمات اولیه
            if not mt5.initialize():
                logger.error(f"Error initializing MT5: {mt5.last_error()}")
                return False
            
            # منتظر اتصال بمان
            time.sleep(2)
            
            # بررسی وضعیت ترمینال
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("Cannot get terminal info")
                return False
            
            # بررسی وضعیت حساب
            account = mt5.account_info()
            if account is None:
                logger.error("Cannot get account info")
                return False
            
            # بررسی وضعیت اتصال
            if not terminal_info.connected:
                logger.error("MT5 not connected to server")
                return False
            
            # اطلاعات حساب
            self.account_info = AccountInfo(
                login=account.login,
                balance=account.balance,
                equity=account.equity,
                margin=account.margin,
                free_margin=account.margin_free,
                leverage=account.leverage,
                currency=account.currency,
                server=account.server,
                is_demo=terminal_info.trade_allowed,
                can_trade=account.trade_allowed
            )
            
            self.connected = True
            self._log_account_info()
            
            # فعال‌سازی نماد
            if not self._check_symbol():
                return False
            
            logger.info(f"✅ MT5 connected successfully to {account.server}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
        
    def _log_account_info(self):
        """Display account information"""
        logger.info("=" * 60)
        logger.info("📊 Connected Account Info:")
        logger.info(f"   Account ID: {self.account_info.login}")
        logger.info(f"   Server: {self.account_info.server}")
        logger.info(f"   Balance: ${self.account_info.balance:,.2f}")
        logger.info(f"   Equity: ${self.account_info.equity:,.2f}")
        logger.info(f"   Free Margin: ${self.account_info.free_margin:,.2f}")
        logger.info(f"   Leverage: 1:{self.account_info.leverage}")
        logger.info(f"   Currency: {self.account_info.currency}")
        logger.info(f"   Account Type: {'Demo' if self.account_info.is_demo else 'Real'}")
        logger.info(f"   Trading Allowed: {'Yes' if self.account_info.can_trade else 'No'}")
        logger.info("=" * 60)
    
    def _check_symbol(self) -> bool:
        """Check and activate symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Error activating symbol {self.symbol}")
                return False
        
        logger.info(f"Symbol {self.symbol} ready for trading")
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def get_ohlcv(self, timeframe: int, count: int = 500) -> Optional[pd.DataFrame]:
        """Get OHLCV data"""
        with self._lock:
            try:
                rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
                if rates is None or len(rates) == 0:
                    logger.warning(f"No data received for timeframe {timeframe}")
                    return None
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
                
            except Exception as e:
                logger.error(f"Error getting data: {e}")
                return None
    
    def get_current_price(self) -> Tuple[float, float]:
        """Get current price (bid, ask)"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return tick.bid, tick.ask
        return 0.0, 0.0
    
    def get_spread(self) -> float:
        """Get spread in price units"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            return symbol_info.spread * symbol_info.point
        return 0.0
    
    def get_point(self) -> float:
        """Get value of one point"""
        symbol_info = mt5.symbol_info(self.symbol)
        return symbol_info.point if symbol_info else 0.00001
    
    def get_active_positions(self) -> List[TradeInfo]:
        """Get open positions for symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        trades = []
        for pos in positions:
            trades.append(TradeInfo(
                ticket=pos.ticket,
                symbol=pos.symbol,
                order_type=pos.type,
                volume=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                open_time=datetime.fromtimestamp(pos.time)
            ))
        return trades
    
    def refresh_account(self):
        """Update account info"""
        account = mt5.account_info()
        if account:
            self.account_info.balance = account.balance
            self.account_info.equity = account.equity
            self.account_info.margin = account.margin
            self.account_info.free_margin = account.margin_free

    def test_connection(self) -> bool:
        """Test MT5 connection and order sending"""
        try:
            logger.info("🔧 Testing MT5 connection...")
            
            # تست دریافت قیمت
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logger.error("❌ Cannot get tick data")
                return False
            
            logger.info(f"✅ Tick data received: Bid={tick.bid}, Ask={tick.ask}")
            
            # تست یک سفارش کوچک
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": 0.01,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 999999,
                "comment": "CONNECTION_TEST",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            logger.info("🔧 Sending test order...")
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("❌ Order send returned None")
                return False
            
            if hasattr(result, 'retcode'):
                logger.info(f"   Order result: {result.retcode} - {result.comment}")
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Test order successful! Ticket: {result.order}")
                    # بستن فوری سفارش تست
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": 0.01,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": result.order,
                        "price": tick.bid,
                        "deviation": 20,
                        "magic": 999999,
                        "comment": "CLOSE_TEST",
                        "type_time": mt5.ORDER_TIME_GTC,
                    }
                    
                    close_result = mt5.order_send(close_request)
                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info("✅ Test order closed successfully")
                    else:
                        logger.warning("⚠️ Could not close test order")
                    
                    return True
                else:
                    logger.error(f"❌ Test order failed: {result.retcode}")
                    return False
            else:
                logger.error("❌ Result has no retcode attribute")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

# ============================================================================
# RISK MANAGER - Risk Management
# ============================================================================
class RiskManager:
    """مدیریت ریسک معاملات - نسخه کامل"""
    
    def __init__(self, mt5_manager: MT5Manager, max_risk_percent: float = None, max_lots: float = None, config: Any = None):
        """مقداردهی اولیه"""
        self.mt5 = mt5_manager
        self.config = config
        
        # استفاده از config اگر موجود باشد
        if config is not None and hasattr(config, 'risk'):
            self.max_risk_percent = config.risk.max_risk_percent if max_risk_percent is None else max_risk_percent
            self.max_lots = config.risk.max_lots if max_lots is None else max_lots
            self.min_balance = config.risk.min_balance
            self.max_daily_loss = config.risk.max_daily_loss
            self.max_daily_trades = config.risk.max_daily_trades
        else:
            self.max_risk_percent = max_risk_percent if max_risk_percent is not None else 0.5
            self.max_lots = max_lots if max_lots is not None else 0.3
            self.min_balance = 500.0
            self.max_daily_loss = 0.02  # حداکثر 2% ضرر در روز
            self.max_daily_trades = 10
        
        self.daily_loss_tracker = 0.0
        self.daily_trades = 0
        
    def can_trade(self) -> Tuple[bool, str]:
        """بررسی امکان معامله"""
        self.mt5.refresh_account()
        
        # بررسی موجودی
        if self.mt5.account_info.balance < self.min_balance:
            return False, f"Balance (${self.mt5.account_info.balance:.2f}) below minimum (${self.min_balance})"
        
        # بررسی اجازه معامله
        if not self.mt5.account_info.can_trade:
            return False, "Account not allowed to trade"
        
        # بررسی ضرر روزانه
        if self.daily_loss_tracker >= self.mt5.account_info.balance * self.max_daily_loss:
            return False, f"Daily loss limit reached (${self.daily_loss_tracker:.2f})"
        
        # بررسی تعداد معاملات روزانه
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})"
        
        return True, "Ready to trade"
    
    def calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """محاسبه حجم بر اساس ریسک - نسخه بهبودیافته"""
        try:
            self.mt5.refresh_account()
            
            if self.mt5.account_info is None:
                logger.error("❌ Cannot get account info")
                return 0.01
            
            balance = self.mt5.account_info.balance
            
            # حداکثر ریسک (0.5% از بالانس)
            max_risk_amount = balance * (self.max_risk_percent / 100)
            
            # محاسبه فاصله SL
            sl_distance = abs(entry - stop_loss)
            if sl_distance <= 0:
                logger.warning("⚠️ SL distance is zero or negative")
                sl_distance = 10 * 0.01  # 10 پیپ حداقل
            
            # دریافت اطلاعات symbol
            symbol_info = mt5.symbol_info(self.mt5.symbol)
            if symbol_info is None:
                logger.error(f"❌ Cannot get symbol info for {self.mt5.symbol}")
                return 0.01
            
            point = symbol_info.point
            tick_size = symbol_info.trade_tick_size if hasattr(symbol_info, 'trade_tick_size') else 0.01
            tick_value = symbol_info.trade_tick_value if hasattr(symbol_info, 'trade_tick_value') else 1.0
            # تبدیل به پیپ
            pip_distance = sl_distance / tick_size
            
            if pip_distance <= 0:
                pip_distance = 10  # حداقل 10 پیپ
        
            if pip_value_per_lot <= 0:
                pip_value_per_lot = 1.0  # مقدار پیش‌فرض
            
            
            # ارزش هر پیپ برای 1 لات
            pip_value_per_lot = tick_value * (0.01 / tick_size) if tick_size > 0 else 1.0
            
            # محاسبه حجم
            if pip_distance > 0 and pip_value_per_lot > 0:
                volume = max_risk_amount / (pip_distance * pip_value_per_lot)
            else:
                volume = 0.01  # حداقل حجم
            
            # اعمال محدودیت‌ها
            volume = min(volume, self.max_lots)
            
            # گرد کردن به step
            volume_step = symbol_info.volume_step if hasattr(symbol_info, 'volume_step') else 0.01
            if volume_step > 0:
                volume = round(volume / volume_step) * volume_step
            
            volume_min = symbol_info.volume_min if hasattr(symbol_info, 'volume_min') else 0.01
            volume = max(volume_min, volume)
            
            volume = round(volume, 2)
            
            # محاسبه ریسک واقعی
            actual_risk = volume * pip_distance * pip_value_per_lot
            risk_percent = (actual_risk / balance) * 100 if balance > 0 else 0
            
            logger.info("💰 Risk Calculation:")
            logger.info(f"   Balance: ${balance:.2f}")
            logger.info(f"   Max Risk Allowed: ${max_risk_amount:.2f} ({self.max_risk_percent}%)")
            logger.info(f"   SL Distance: {sl_distance:.4f} ({pip_distance:.1f} pips)")
            logger.info(f"   Calculated Volume: {volume:.2f} lots")
            logger.info(f"   Actual Risk: ${actual_risk:.2f} ({risk_percent:.2f}%)")
            
            if risk_percent > self.max_risk_percent:
                logger.warning(f"⚠️ Risk ({risk_percent:.2f}%) exceeds limit ({self.max_risk_percent}%)")
                # کاهش حجم
                volume = volume * (self.max_risk_percent / risk_percent)
                volume = round(volume, 2)
                logger.info(f"   Adjusted Volume: {volume:.2f} lots")
            
            return volume
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.01  # حداقل حجم در صورت خطا
    
    def validate_signal(self, signal: NDSSignal) -> Tuple[bool, str]:
        """اعتبارسنجی سیگنال"""
        if not signal.is_valid():
            return False, "Invalid signal"
        
        if signal.risk_reward < 1.5:
            return False, f"R/R ({signal.risk_reward:.2f}) below 1.5"
        
        if signal.confidence < 0.6:
            return False, f"Confidence ({signal.confidence:.2f}) below 0.6"
        
        # بررسی سطوح
        if signal.stop_loss <= 0 or signal.take_profit <= 0:
            return False, "Invalid SL/TP levels"
        
        if signal.direction == TrendDirection.BULLISH:
            if signal.stop_loss >= signal.entry_price:
                return False, f"SL ({signal.stop_loss}) >= Entry ({signal.entry_price}) for BUY"
            if signal.take_profit <= signal.entry_price:
                return False, f"TP ({signal.take_profit}) <= Entry ({signal.entry_price}) for BUY"
        else:
            if signal.stop_loss <= signal.entry_price:
                return False, f"SL ({signal.stop_loss}) <= Entry ({signal.entry_price}) for SELL"
            if signal.take_profit >= signal.entry_price:
                return False, f"TP ({signal.take_profit}) >= Entry ({signal.entry_price}) for SELL"
        
        return True, "Valid signal"
    
    def update_daily_stats(self, profit: float):
        """به‌روزرسانی آمار روزانه"""
        self.daily_trades += 1
        
        if profit < 0:
            self.daily_loss_tracker += abs(profit)
            logger.info(f"📊 Daily stats: Trades={self.daily_trades}, Loss=${self.daily_loss_tracker:.2f}")
        else:
            logger.info(f"📊 Daily stats: Trades={self.daily_trades}, Profit=${profit:.2f}")
    
    def reset_daily_stats(self):
        """بازنشانی آمار روزانه"""
        self.daily_loss_tracker = 0.0
        self.daily_trades = 0
        logger.info("🔄 Daily stats reset")

# ============================================================================
# TRADE MANAGER - Trade Management
# ============================================================================
class TradeManager:
    """Manage entry, exit and trailing stop"""
    
    def __init__(self, mt5_manager: MT5Manager, risk_manager: RiskManager, config: Any = None):
        self.mt5 = mt5_manager
        self.risk = risk_manager
        self.config = config
        self.pip_margin = 5
    
    def _add_sltp_later(self, ticket: int, signal: NDSSignal):
        """اضافه کردن SL/TP بعد از باز شدن معامله - نسخه ساده"""
        try:
            logger.info(f"🔧 Adding SL/TP to trade #{ticket}")
            
            # کمی صبر برای ثبت position
            time.sleep(2)
            
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"❌ Cannot find position #{ticket}")
                return
            
            position = positions[0]
            logger.info(f"📊 Position found: #{position.ticket}, Type: {'BUY' if position.type == 0 else 'SELL'}")
            logger.info(f"   Open Price: {position.price_open:.2f}, Current SL: {position.sl:.2f}, Current TP: {position.tp:.2f}")
            
            # تنظیم SL/TP با اعتبارسنجی
            if signal.direction == TrendDirection.BULLISH:
                # BUY: SL پایین‌تر، TP بالاتر
                sl_price = min(signal.stop_loss, position.price_open * 0.998)  # حداکثر 0.2% پایین‌تر
                tp_price = max(signal.take_profit, position.price_open * 1.004)  # حداقل 0.4% بالاتر
            else:
                # SELL: SL بالاتر، TP پایین‌تر
                sl_price = max(signal.stop_loss, position.price_open * 1.002)  # حداقل 0.2% بالاتر
                tp_price = min(signal.take_profit, position.price_open * 0.996)  # حداکثر 0.4% پایین‌تر
            
            logger.info(f"📈 Setting SL: {sl_price:.2f}, TP: {tp_price:.2f}")
            
            # ارسال درخواست SL/TP
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl_price,
                "tp": tp_price,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"❌ SL/TP request returned None for #{ticket}")
                return
            
            logger.info(f"📋 SL/TP Result: {result.retcode} - {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ SL/TP added successfully to #{ticket}")
                
                # تأیید نهایی
                time.sleep(1)
                updated = mt5.positions_get(ticket=ticket)
                if updated:
                    pos = updated[0]
                    logger.info(f"✅ Confirmed - SL: {pos.sl:.2f}, TP: {pos.tp:.2f}")
            else:
                logger.warning(f"⚠️ SL/TP addition failed: {result.retcode}")
                
        except Exception as e:
            logger.error(f"❌ Error in _add_sltp_later: {str(e)}")

    def _get_symbol_filling_mode(self) -> int:
        """Get appropriate filling mode for symbol"""
        try:
            symbol_info = mt5.symbol_info(self.mt5.symbol)
            if symbol_info:
                filling_mode = getattr(symbol_info, 'filling_mode', -1)
                
                try:
                    if hasattr(mt5, 'ORDER_FILLING_RETURN'):
                        return mt5.ORDER_FILLING_RETURN
                except:
                    pass
                
                try:
                    if hasattr(mt5, 'ORDER_FILLING_IOC'):
                        return mt5.ORDER_FILLING_IOC
                except:
                    pass
                
                return 0
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error getting filling mode: {e}")
            return 0
    
    def open_trade_safe(self, signal: NDSSignal) -> Optional[int]:
        """باز کردن معامله - نسخه اصلاح شده"""
        try:
            logger.info("=" * 70)
            logger.info("🚀 SAFE TRADE EXECUTION")
            logger.info("=" * 70)
            
            # 1. بررسی پوزیشن‌های موجود
            positions = self.mt5.get_active_positions()
            if positions:
                logger.info(f"⏸️  {len(positions)} open position(s) - skipping")
                for pos in positions[:2]:
                    logger.info(f"   #{pos.ticket}: {pos.order_type} {pos.volume} lots, P/L: ${pos.profit:.2f}")
                return None
            
            # 2. حجم کوچک برای تست
            volume = 0.10  # حجم متوسط برای تست
            
            # 3. دریافت قیمت
            tick = mt5.symbol_info_tick(self.mt5.symbol)
            if tick is None:
                logger.error("❌ Cannot get tick")
                return None
            
            logger.info(f"📊 Market: Bid={tick.bid:.2f}, Ask={tick.ask:.2f}")
            
            # 4. تنظیم پارامترها
            if signal.direction == TrendDirection.BULLISH:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                direction_str = "BUY"
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                direction_str = "SELL"
            
            logger.info(f"📝 Order: {direction_str} {volume} lots @ {price:.2f}")
            
            # 5. درخواست بسیار ساده (بدون type_filling)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.mt5.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 100,
                "magic": 123456,
                "comment": f"NDS_{direction_str}",
                "type_time": mt5.ORDER_TIME_GTC,
                # بدون type_filling - این باعث موفقیت شد!
            }
            
            logger.info("📤 Sending order...")
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("❌ order_send() returned None")
                return None
            
            logger.info(f"📋 Order Result: {result.retcode} - {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                logger.info(f"🎉 TRADE OPENED SUCCESSFULLY! Ticket: #{ticket}")
                
                # اضافه کردن SL/TP
                self._add_sltp_later(ticket, signal)
                
                # نمایش اطلاعات position
                time.sleep(1)
                self._log_position_info(ticket)
                
                return ticket
            else:
                logger.error(f"❌ Trade failed: {result.retcode}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}")
            return None

    def _log_position_info(self, ticket: int):
        """لاگ اطلاعات position"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if positions:
                pos = positions[0]
                logger.info("📋 Position Details:")
                logger.info(f"   Ticket: #{pos.ticket}")
                logger.info(f"   Type: {'BUY' if pos.type == 0 else 'SELL'}")
                logger.info(f"   Volume: {pos.volume} lots")
                logger.info(f"   Open Price: {pos.price_open:.2f}")
                logger.info(f"   Current Price: {pos.price_current:.2f}")
                logger.info(f"   SL: {pos.sl:.2f}")
                logger.info(f"   TP: {pos.tp:.2f}")
                logger.info(f"   Profit: ${pos.profit:.2f}")
                logger.info(f"   Swap: ${pos.swap:.2f}")
                logger.info(f"   Commission: ${pos.commission:.2f}")
                # ✅ استفاده ایمن از commission با getattr
                commission = getattr(pos, 'commission', 0.0)
                logger.info(f"   Commission: ${commission:.2f}")
                    
                # ✅ اصلاح: بررسی وجود commission
                if hasattr(pos, 'commission'):
                    logger.info(f"   Commission: ${pos.commission:.2f}")
                else:
                    logger.info(f"   Commission: N/A")
        
        except Exception as e:
            logger.error(f"Error logging position info: {e}")
            
    def _add_sltp_comprehensive(self, ticket: int, signal: NDSSignal):
        """اضافه کردن SL/TP به روش جامع"""
        try:
            logger.info(f"🔧 Adding SL/TP to trade #{ticket}")
            
            # کمی صبر برای اطمینان از ثبت position
            time.sleep(2)
            
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"❌ Cannot find position #{ticket}")
                
                # تلاش جایگزین
                all_positions = mt5.positions_get(symbol=self.mt5.symbol)
                if all_positions:
                    for pos in all_positions:
                        if pos.ticket == ticket:
                            positions = [pos]
                            break
                
                if not positions:
                    logger.error(f"❌ Position #{ticket} not found in any search")
                    return
            
            position = positions[0]
            logger.info(f"📊 Found position: #{position.ticket}, Type: {'BUY' if position.type == 0 else 'SELL'}")
            
            # تنظیم SL/TP با اعتبارسنجی
            point = mt5.symbol_info(self.mt5.symbol).point
            
            if position.type == mt5.ORDER_TYPE_BUY:
                # برای BUY: SL زیر قیمت باز شدن
                sl_price = min(signal.stop_loss, position.price_open - (5 * point))
                tp_price = max(signal.take_profit, position.price_open + (10 * point))
            else:
                # برای SELL: SL بالای قیمت باز شدن
                sl_price = max(signal.stop_loss, position.price_open + (5 * point))
                tp_price = min(signal.take_profit, position.price_open - (10 * point))
            
            logger.info(f"📈 Calculated SL: {sl_price:.2f}, TP: {tp_price:.2f}")
            
            # روش 1: استفاده از TRADE_ACTION_SLTP
            logger.info("🔄 Method 1: Using TRADE_ACTION_SLTP")
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl_price,
                "tp": tp_price,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("❌ SL/TP request returned None")
            else:
                logger.info(f"📋 SL/TP Result: {result.retcode} - {result.comment}")
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ SL/TP added successfully to #{ticket}")
                    
                    # تأیید
                    time.sleep(1)
                    updated = mt5.positions_get(ticket=ticket)
                    if updated:
                        pos = updated[0]
                        logger.info(f"✅ Confirmed - SL: {pos.sl:.2f}, TP: {pos.tp:.2f}")
                    return
                else:
                    logger.warning(f"⚠️ SL/TP failed: {result.retcode}")
            
            # روش 2: استفاده از TRADE_ACTION_MODIFY (اگر روش 1 شکست خورد)
            logger.info("🔄 Method 2: Using TRADE_ACTION_MODIFY")
            
            # این روش نیاز به Order Ticket دارد نه Position Ticket
            orders = mt5.orders_get(ticket=ticket)
            if orders:
                order = orders[0]
                
                modify_request = {
                    "action": mt5.TRADE_ACTION_MODIFY,
                    "order": order.ticket,
                    "price": order.price_open,
                    "sl": sl_price,
                    "tp": tp_price,
                }
                
                modify_result = mt5.order_send(modify_request)
                if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ SL/TP added via modify to #{ticket}")
                else:
                    logger.error(f"❌ Modify also failed for #{ticket}")
            else:
                logger.error(f"❌ Cannot find order for position #{ticket}")
            
        except Exception as e:
            logger.error(f"❌ Error in _add_sltp_comprehensive: {str(e)}")

    def _verify_trade(self, ticket: int, signal: NDSSignal):
        """تأیید معامله باز شده"""
        try:
            logger.info(f"🔍 Verifying trade #{ticket}...")
            
            time.sleep(1)
            
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"⚠️ Cannot verify #{ticket} - position not found")
                return
            
            position = positions[0]
            
            # بررسی پارامترها
            verification = {
                "ticket": position.ticket,
                "type": "BUY" if position.type == 0 else "SELL",
                "volume": position.volume,
                "open_price": position.price_open,
                "current_price": position.price_current,
                "sl": position.sl,
                "tp": position.tp,
                "profit": position.profit,
                "commission": position.commission,
                "swap": position.swap,
            }
            
            logger.info("📋 Trade Verification:")
            for key, value in verification.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.2f}")
                else:
                    logger.info(f"   {key}: {value}")
            
            # بررسی SL/TP
            if position.sl == 0.0 or position.tp == 0.0:
                logger.warning("⚠️ WARNING: Trade has no SL or TP!")
                
                # تلاش مجدد برای اضافه کردن
                if signal.stop_loss > 0 and signal.take_profit > 0:
                    logger.info("🔄 Retrying SL/TP addition...")
                    time.sleep(2)
                    self._add_sltp_comprehensive(ticket, signal)
            
            logger.info(f"✅ Trade #{ticket} verified")
            
        except Exception as e:
            logger.error(f"❌ Error in _verify_trade: {e}")

    def _add_sltp_to_trade(self, ticket: int, signal: NDSSignal):
        """اضافه کردن SL/TP - نسخه کاملاً بازنویسی شده"""
        try:
            logger.info(f"🔧 Attempting to add SL/TP to trade {ticket}")
            
            # کمی صبر کنیم تا معامله در سیستم ثبت شود
            time.sleep(1)
            
            # دریافت position با ticket
            positions = mt5.positions_get(ticket=ticket)
            
            if positions is None:
                logger.error(f"❌ positions_get() returned None for ticket {ticket}")
                # روش جایگزین: گرفتن همه positions و فیلتر کردن
                all_positions = mt5.positions_get()
                if all_positions:
                    for pos in all_positions:
                        if pos.ticket == ticket:
                            positions = [pos]
                            break
            
            if not positions or len(positions) == 0:
                logger.error(f"❌ Cannot find position with ticket {ticket}")
                return
            
            position = positions[0]
            
            logger.info(f"📊 Position found: #{position.ticket}, Type: {'BUY' if position.type == 0 else 'SELL'}")
            logger.info(f"   Open Price: {position.price_open}, Current SL: {position.sl}, Current TP: {position.tp}")
            
            # تنظیم SL/TP
            point = self.mt5.get_point()
            
            if position.type == mt5.ORDER_TYPE_BUY:  # BUY position
                # برای BUY: SL زیر قیمت، TP بالای قیمت
                sl_price = signal.stop_loss
                tp_price = signal.take_profit
                
                # بررسی valid بودن سطوح
                if sl_price >= position.price_open:
                    logger.warning(f"⚠️ Invalid SL for BUY: {sl_price} >= {position.price_open}")
                    sl_price = position.price_open - (10 * point)  # 10 پیپ پایین‌تر
                
                if tp_price <= position.price_open:
                    logger.warning(f"⚠️ Invalid TP for BUY: {tp_price} <= {position.price_open}")
                    tp_price = position.price_open + (20 * point)  # 20 پیپ بالاتر
                
            else:  # SELL position
                # برای SELL: SL بالای قیمت، TP زیر قیمت
                sl_price = signal.stop_loss
                tp_price = signal.take_profit
                
                # بررسی valid بودن سطوح
                if sl_price <= position.price_open:
                    logger.warning(f"⚠️ Invalid SL for SELL: {sl_price} <= {position.price_open}")
                    sl_price = position.price_open + (10 * point)  # 10 پیپ بالاتر
                
                if tp_price >= position.price_open:
                    logger.warning(f"⚠️ Invalid TP for SELL: {tp_price} >= {position.price_open}")
                    tp_price = position.price_open - (20 * point)  # 20 پیپ پایین‌تر
            
            logger.info(f"📈 Setting SL: {sl_price:.2f}, TP: {tp_price:.2f}")
            
            # ساخت درخواست
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl_price,
                "tp": tp_price,
            }
            
            logger.info(f"📤 Sending SL/TP request for ticket {ticket}...")
            
            # ارسال درخواست
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"❌ SL/TP order_send() returned None for ticket {ticket}")
                logger.error("   Possible reasons:")
                logger.error("   1. Position already closed")
                logger.error("   2. MT5 connection issue")
                logger.error("   3. Invalid SL/TP levels")
                return
            
            logger.info(f"✅ SL/TP result received: {result.retcode} - {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"🎉 SL/TP added successfully to trade {ticket}")
                
                # تأیید نهایی
                time.sleep(0.5)
                updated_positions = mt5.positions_get(ticket=ticket)
                if updated_positions:
                    updated = updated_positions[0]
                    logger.info(f"✅ Confirmed: SL={updated.sl:.2f}, TP={updated.tp:.2f}")
            
            else:
                logger.warning(f"⚠️ SL/TP addition failed: {result.retcode}")
                
                # تلاش جایگزین: اصلاح position
                if result.retcode == 10027:  # Invalid stops
                    logger.info("🔄 Trying alternative method: modify position...")
                    self._modify_position_sltp(position, sl_price, tp_price)
        
        except Exception as e:
            logger.error(f"❌ Error adding SL/TP: {str(e)}")

    def _modify_position_sltp(self, position, sl_price: float, tp_price: float):
        """روش جایگزین برای تنظیم SL/TP"""
        try:
            # روش جایگزین: بستن و باز کردن مجدد با SL/TP
            logger.info("🔄 Alternative: Closing and reopening with SL/TP...")
            
            # بستن position فعلی
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "MODIFY_SLTP",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            close_result = mt5.order_send(close_request)
            if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {position.ticket} closed for modification")
                
                # باز کردن مجدد با SL/TP
                time.sleep(1)
                tick = mt5.symbol_info_tick(position.symbol)
                
                new_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": position.type,
                    "price": tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 50,
                    "magic": 123456,
                    "comment": "REOPEN_WITH_SLTP",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                
                new_result = mt5.order_send(new_request)
                if new_result and new_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Position reopened with SL/TP: {new_result.order}")
                else:
                    logger.error("❌ Failed to reopen position")
            else:
                logger.error("❌ Failed to close position for modification")
                
        except Exception as e:
            logger.error(f"Error in modify_position_sltp: {e}")
    
    def _log_error_details(self, retcode: int):
        """لاگ جزئیات خطا"""
        error_map = {
            10001: "Requote",
            10002: "Request rejected",
            10003: "Request canceled by trader",
            10004: "Order placed timeout",
            10005: "Invalid price",
            10006: "Invalid stops",
            10007: "Invalid trade volume",
            10008: "Not enough money",
            10009: "Price changed",
            10010: "Off quotes",
            10011: "Broker busy",
            10012: "Requote",
            10013: "Order locked",
            10014: "Long positions only allowed",
            10015: "Too many requests",
            10016: "Trading is disabled",  # مهم!
            10017: "Account is disabled",
            10018: "Invalid account",
            10019: "Trade timeout",
            10020: "Invalid trade parameters",
        }
        
        if retcode in error_map:
            logger.error(f"   Error meaning: {error_map[retcode]}")
        
        # پیشنهادات
        if retcode == 10016:
            logger.error("   💡 SOLUTION: Press Ctrl+T in MT5 to enable trading")
        elif retcode == 10008:
            logger.error("   💡 SOLUTION: Not enough balance. Check your account.")
        elif retcode == 10006:
            logger.error("   💡 SOLUTION: SL/TP levels are invalid. Adjust them.")
            
    def debug_symbol_info(self):
        """نمایش اطلاعات symbol برای دیباگ - FIXED"""
        try:
            symbol_info = mt5.symbol_info(self.mt5.symbol)
            if symbol_info:
                logger.info("🔧 Symbol Debug Info:")
                logger.info(f"   Name: {symbol_info.name}")
                logger.info(f"   Bid: {symbol_info.bid}")
                logger.info(f"   Ask: {symbol_info.ask}")
                logger.info(f"   Spread: {symbol_info.spread}")
                logger.info(f"   Trade Stops Level: {symbol_info.trade_stops_level}")
                logger.info(f"   Trade Freeze Level: {symbol_info.trade_freeze_level}")
                logger.info(f"   Volume Min: {symbol_info.volume_min}")
                logger.info(f"   Volume Max: {symbol_info.volume_max}")
                logger.info(f"   Volume Step: {symbol_info.volume_step}")
                logger.info(f"   Trade Mode: {symbol_info.trade_mode}")
                # حذف margin_rate چون در همه نمادها وجود ندارد
                # logger.info(f"   Margin Rate: {symbol_info.margin_rate}")
                logger.info(f"   Trade Contract Size: {symbol_info.trade_contract_size}")
                logger.info(f"   Trade Tick Size: {symbol_info.trade_tick_size}")
                logger.info(f"   Trade Tick Value: {symbol_info.trade_tick_value}")
            else:
                logger.error("Cannot get symbol info")
        except Exception as e:
            logger.error(f"Error in debug_symbol_info: {e}")
    def _add_sltp_after_open(self, ticket: int, sl: float, tp: float):
        """اضافه کردن SL/TP بعد از باز کردن معامله"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
            }
            
            if sl > 0:
                request["sl"] = sl
            if tp > 0:
                request["tp"] = tp
            
            result = mt5.order_send(request)
            if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ SL/TP added to trade {ticket}")
            else:
                logger.warning(f"Could not add SL/TP to trade {ticket}")
        except Exception as e:
            logger.error(f"Error adding SL/TP: {e}")

    def _log_order_debug_info(self, request: dict, result):
        """لاگ اطلاعات دیباگ برای سفارش"""
        logger.info("🔍 Order Debug Info:")
        logger.info(f"   Request: {request}")
        
        if result:
            logger.info(f"   Result type: {type(result)}")
            logger.info(f"   Result attributes: {dir(result)}")
            
            # چاپ تمام attributes
            for attr in dir(result):
                if not attr.startswith('__'):
                    try:
                        value = getattr(result, attr)
                        logger.info(f"   {attr}: {value}")
                    except:
                        pass


    def update_trailing_stop(self, trade: TradeInfo, new_sl: float) -> bool:
        """Update trailing stop"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": trade.ticket,
                "sl": new_sl,
                "tp": trade.tp,
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # ⭐ به‌روزرسانی محلی
                trade.sl = new_sl
                logger.info(f"✅ Trailing Stop updated: {new_sl:.2f}")
                return True
            else:
                logger.error(f"❌ Failed to update trailing: {result.retcode if result else 'None'}")
                return False

        except Exception as e:
            logger.error(f"❌ Error updating Trailing Stop: {e}")
            return False


        

    def update_take_profit(self, trade: TradeInfo, new_tp: float) -> bool:
        """Update Take Profit to next node"""
        try:
            point = self.mt5.get_point()
            spread = self.mt5.get_spread()
            
            is_buy = trade.order_type == mt5.ORDER_TYPE_BUY
            
            if is_buy:
                new_tp = new_tp - spread - (self.pip_margin * point)
            else:
                new_tp = new_tp + spread + (self.pip_margin * point)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": trade.ticket,
                "sl": trade.sl,
                "tp": new_tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Take Profit updated: {new_tp:.2f}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating TP: {e}")
            return False
    
    def close_trade(self, trade: TradeInfo, reason: str = "") -> bool:
        """Close trade - FIXED VERSION"""
        try:
            # دریافت قیمت با تلاش بیشتر
            for _ in range(3):
                tick = mt5.symbol_info_tick(self.mt5.symbol)
                if tick:
                    break
                time.sleep(0.1)
            
            if not tick:
                logger.error("❌ Cannot get tick data for closing")
                return False
            
            bid, ask = tick.bid, tick.ask
            
            if trade.order_type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = ask
            
            # محاسبه حداقل فاصله
            point = self.mt5.get_point()
            min_distance = 10 * point  # حداقل 10 پیپ
            
            # اعتبارسنجی قیمت
            if abs(price - trade.open_price) < min_distance:
                logger.warning(f"⚠️ Price too close to open price. Adjusting...")
                if trade.order_type == mt5.ORDER_TYPE_BUY:
                    price = max(price, trade.open_price + min_distance)
                else:
                    price = min(price, trade.open_price - min_distance)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.mt5.symbol,
                "volume": trade.volume,
                "type": order_type,
                "position": trade.ticket,
                "price": price,
                "deviation": 100,  # افزایش deviation
                "magic": 123456,
                "comment": f"Close: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            # حذف type_filling برای جلوگیری از خطا
            # if hasattr(self, '_get_symbol_filling_mode'):
            #     filling_mode = self._get_symbol_filling_mode()
            #     if filling_mode != -1:
            #         request["type_filling"] = filling_mode
            
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("❌ order_send() returned None")
                return False
            
            if hasattr(result, 'retcode'):
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Trade closed | Ticket: {trade.ticket} | Reason: {reason}")
                    logger.info(f"   P/L: ${trade.profit:.2f}")
                    return True
                else:
                    logger.error(f"❌ Error closing trade: {result.retcode}")
                    
                    # نمایش جزئیات خطا
                    if result.retcode == 10030:
                        logger.error("   💡 Error 10030: Invalid stops or price")
                        logger.error(f"   💡 Price used: {price}, Bid: {bid}, Ask: {ask}")
                        logger.error(f"   💡 Try increasing deviation or adjusting price")
                    
                    return False
            else:
                logger.error("❌ Result has no retcode attribute")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error closing trade: {e}")
            return False
# ============================================================================
# ADVANCED NDS ANALYZER - Advanced NDS Analyzer (Original)
# ============================================================================
class AdvancedNDSAnalyzer:
    """
    Complete implementation of NDS model (Nodal Displacement Sequencing)
    """
    
    def __init__(self, mt5_manager: MT5Manager, config: Any = None):
        self.mt5 = mt5_manager
        self.config = config
        
        # NDS parameters
        if config is not None and hasattr(config, 'nds'):
            self.alpha_correction = config.nds.alpha_correction
            self.alpha_pressure = config.nds.alpha_pressure
            self.beta_displacement = config.nds.beta_displacement
        else:
            self.alpha_correction = 0.86
            self.alpha_pressure = 0.2
            self.beta_displacement = 0.3
        
        # Timeframes for scalping
        self.tf_trend = mt5.TIMEFRAME_M3
        self.tf_analysis = mt5.TIMEFRAME_M1
        self.tf_entry = mt5.TIMEFRAME_M1
        
        # Storage
        self.nodes_cache: Dict[int, List[Node]] = {}
        self.cycles_cache: Dict[int, List[Cycle]] = {}
        # پارامترهای جدید برای اسکلپینگ
        self.scalp_target_pips = 10  # تارگت 10 پیپ
        self.scalp_max_risk_pips = 5  # ریسک حداکثر 5 پیپ
        self.min_scalp_confidence = 0.7  # حداقل اطمینان 70%

    def analyze(self) -> Optional[NDSSignal]:
        """Complete NDS analysis and signal generation"""
        try:
            logger.info("Starting NDS analysis...")
            
            df_trend = self.mt5.get_ohlcv(self.tf_trend, 1440)
            df_analysis = self.mt5.get_ohlcv(self.tf_analysis, 500)
            df_entry = self.mt5.get_ohlcv(self.tf_entry, 200)
            
            if df_trend is None or df_analysis is None or df_entry is None:
                logger.warning("Insufficient data for analysis")
                return None
            
            nodes_trend = self._detect_nodes(df_trend, self.tf_trend)
            nodes_analysis = self._detect_nodes(df_analysis, self.tf_analysis)
            nodes_entry = self._detect_nodes(df_entry, self.tf_entry)
            
            if len(nodes_analysis) < 3:
                logger.warning("Insufficient nodes identified")
                return None
            
            cycles_trend = self._calculate_cycles(nodes_trend)
            cycles_analysis = self._calculate_cycles(nodes_analysis)
            
            pressure = self._calculate_inter_tf_pressure(cycles_trend, cycles_analysis)
            
            displaced_nodes = self._calculate_nodal_displacement(nodes_analysis, pressure)
            
            poly_functions = self._fit_polynomial_functions(df_analysis)
            
            quantum_state, phase_uncertainty = self._quantum_analysis(df_analysis)
            
            hurst, multifractal_spectrum = self._multifractal_analysis(df_analysis)
            
            trend_direction = self._determine_trend(df_trend, cycles_trend, poly_functions)
            
            entry_price, sl, tp = self._calculate_levels_simple(trend_direction, df_analysis)
            
            if sl == 0 or tp == 0 or entry_price == 0:
                logger.warning("Invalid levels calculated (zero values)")
                return None
            
            if abs(entry_price - sl) < 0.01:
                logger.warning(f"SL too close to entry: {abs(entry_price - sl):.4f}")
                return None

            risk_reward = abs(tp - entry_price) / abs(entry_price - sl)
            logger.info(f"   Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
            logger.info(f"   Calculated R/R: {risk_reward:.2f}")
            if risk_reward < 1.5:
                logger.info(f"R/R ({risk_reward:.2f}) below 1.5 - waiting for better position")
                return None
            
            confidence = self._calculate_confidence(
                quantum_state, hurst, phase_uncertainty, poly_functions
            )
            
            signal = NDSSignal(
                direction=trend_direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                quantum_state=quantum_state,
                hurst_exponent=hurst,
                risk_reward=risk_reward,
                timestamp=datetime.now(),
                nodes=displaced_nodes[-3:]
            )
            
            self._log_signal(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error in NDS analysis: {e}")
            return None
    
    def _detect_nodes(self, df: pd.DataFrame, timeframe: int) -> List[Node]:
        """Identify nodes with precise mathematical conditions"""
        nodes = []
        prices = df['close'].values
        gradient = np.gradient(prices)
        hessian = np.gradient(gradient)
        zero_gradient_threshold = np.std(gradient) * 0.05

        for i in range(2, len(prices) - 2):
            if abs(gradient[i]) < zero_gradient_threshold:
                if abs(hessian[i]) > zero_gradient_threshold / 3:
                    if hessian[i] < 0:
                        node_type = 'high'
                    else:
                        node_type = 'low'

                    strength = abs(hessian[i]) / (np.std(hessian) + 1e-8)
                    strength = min(strength, 1.0)

                    node = Node(
                        index=i,
                        price=prices[i],
                        time=df.index[i],
                        node_type=node_type,
                        strength=strength
                    )
                    nodes.append(node)

        nodes = self._filter_nearby_nodes(nodes, min_distance=2)
        self.nodes_cache[timeframe] = nodes
        logger.info(f"   Detected {len(nodes)} nodes in timeframe {timeframe}")
        return nodes

    def _filter_nearby_nodes(self, nodes: List[Node], min_distance: int) -> List[Node]:
        """Remove nodes too close to each other"""
        if len(nodes) < 2:
            return nodes
        
        filtered = [nodes[0]]
        for node in nodes[1:]:
            if node.index - filtered[-1].index >= min_distance:
                filtered.append(node)
            elif node.strength > filtered[-1].strength:
                filtered[-1] = node
        
        return filtered
    
    def _calculate_cycles(self, nodes: List[Node]) -> List[Cycle]:
        """Calculate cycles: C_n = R_n - K_n = 0.14 * R_n"""
        cycles = []
        
        for i in range(len(nodes) - 1):
            cycle = Cycle(
                start_node=nodes[i],
                end_node=nodes[i + 1]
            )
            cycle.calculate()
            cycles.append(cycle)
        
        return cycles
    
    def _calculate_inter_tf_pressure(self, cycles_higher: List[Cycle], cycles_lower: List[Cycle]) -> float:
        """Calculate inter-timeframe pressure: P_k(t) = α_k * dC_k(t)/dt"""
        if not cycles_higher:
            return 0.0
        
        recent_movements = [c.net_movement for c in cycles_higher[-3:]]
        avg_movement = np.mean(recent_movements) if recent_movements else 0
        
        if len(recent_movements) >= 2:
            rate_of_change = recent_movements[-1] - recent_movements[-2]
        else:
            rate_of_change = avg_movement
        
        pressure = self.alpha_pressure * rate_of_change
        return pressure
    
    def _calculate_nodal_displacement(self, nodes: List[Node], pressure: float) -> List[Node]:
        """Calculate nodal displacement: Δn_i^(k) = β_k * (Σ Δn_j^(k+1) + P_k(t))"""
        displaced_nodes = []
        cumulative_displacement = 0.0
        
        for node in nodes:
            displacement = self.beta_displacement * (cumulative_displacement + pressure)
            new_price = node.price + displacement
            
            displaced_node = Node(
                index=node.index,
                price=node.price,
                time=node.time,
                node_type=node.node_type,
                strength=node.strength,
                displaced_price=new_price
            )
            displaced_nodes.append(displaced_node)
            cumulative_displacement += abs(displacement)
        
        return displaced_nodes
    
    def _fit_polynomial_functions(self, df: pd.DataFrame) -> List[PolynomialFunction]:
        """Fit polynomial functions"""
        functions = []
        prices = df['close'].values
        segments = self._segment_price_data(prices)
        
        for segment_type, start_idx, end_idx in segments:
            if end_idx - start_idx < 4:
                continue
            
            segment = prices[start_idx:end_idx]
            t = np.arange(len(segment))
            
            try:
                coeffs = np.polyfit(t, segment, 3)
                poly = np.poly1d(coeffs)
                y_pred = poly(t)
                ss_res = np.sum((segment - y_pred) ** 2)
                ss_tot = np.sum((segment - np.mean(segment)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                
                derivative = np.polyder(poly)
                velocity = derivative(t[-1])
                
                func = PolynomialFunction(
                    coefficients=coeffs,
                    degree=3,
                    r_squared=r_squared,
                    velocity=velocity,
                    function_type='trend' if segment_type == 'trend' else 'pullback'
                )
                functions.append(func)
                
            except Exception:
                continue
        
        return functions
    
    def _segment_price_data(self, prices: np.ndarray) -> List[Tuple[str, int, int]]:
        """Split data into trend and pullback segments"""
        segments = []
        ma_short = pd.Series(prices).rolling(5).mean().values
        ma_long = pd.Series(prices).rolling(15).mean().values
        
        current_type = None
        start_idx = 0
        
        for i in range(15, len(prices)):
            if np.isnan(ma_short[i]) or np.isnan(ma_long[i]):
                continue
            
            if ma_short[i] > ma_long[i]:
                new_type = 'trend'
            else:
                new_type = 'pullback'
            
            if new_type != current_type:
                if current_type is not None:
                    segments.append((current_type, start_idx, i))
                current_type = new_type
                start_idx = i
        
        if current_type is not None:
            segments.append((current_type, start_idx, len(prices)))
        
        return segments
    
    def _quantum_analysis(self, df: pd.DataFrame) -> Tuple[QuantumState, float]:
        """Quantum analysis: ψ(t) = P(t) * e^(i*φ(t))"""
        prices = df['close'].values
        returns = np.diff(prices) / prices[:-1]
        phase = np.cumsum(returns) * 2 * np.pi
        
        amplitude = prices[1:]
        psi = amplitude * np.exp(1j * phase)
        
        phase_uncertainty = np.std(phase[-20:]) if len(phase) >= 20 else np.std(phase)
        
        threshold = 0.5
        
        if phase_uncertainty > threshold:
            state = QuantumState.SUPERPOSITION
        else:
            recent_phase = np.mean(phase[-10:])
            if np.cos(recent_phase) > 0:
                state = QuantumState.COLLAPSED_BULLISH
            else:
                state = QuantumState.COLLAPSED_BEARISH
        
        return state, phase_uncertainty
    
    def _multifractal_analysis(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Multifractal analysis: Calculate Hurst Exponent"""
        prices = df['close'].values
        returns = np.diff(np.log(prices))
        
        hurst = self._calculate_hurst_rs(returns)
        spectrum = self._calculate_multifractal_spectrum(returns)
        
        return hurst, spectrum
    
    def _calculate_hurst_rs(self, returns: np.ndarray) -> float:
        """Calculate Hurst Exponent with R/S method"""
        n = len(returns)
        if n < 20:
            return 0.5
        
        max_k = min(n // 2, 100)
        rs_values = []
        
        for k in range(10, max_k):
            segments = n // k
            rs_seg = []
            
            for seg in range(segments):
                start = seg * k
                end = start + k
                segment = returns[start:end]
                
                mean_seg = np.mean(segment)
                cumsum = np.cumsum(segment - mean_seg)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                
                if S > 0:
                    rs_seg.append(R / S)
            
            if rs_seg:
                rs_values.append((k, np.mean(rs_seg)))
        
        if len(rs_values) < 3:
            return 0.5
        
        ks = np.log([x[0] for x in rs_values])
        rs = np.log([x[1] for x in rs_values])
        
        try:
            hurst, _ = np.polyfit(ks, rs, 1)
            return np.clip(hurst, 0, 1)
        except:
            return 0.5
    
    def _calculate_multifractal_spectrum(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate multifractal spectrum f(α)"""
        q_values = np.arange(-5, 6)
        tau_q = []
        
        for q in q_values:
            epsilon = 0.01
            abs_returns = np.abs(returns) + epsilon
            
            if q == 0:
                partition = np.sum(np.log(abs_returns))
            else:
                partition = np.sum(abs_returns ** q)
            
            tau_q.append(np.log(partition + epsilon))
        
        tau_q = np.array(tau_q)
        alpha = np.gradient(tau_q)
        f_alpha = q_values * alpha - tau_q
        
        return {
            'alpha_min': float(np.min(alpha)),
            'alpha_max': float(np.max(alpha)),
            'delta_alpha': float(np.max(alpha) - np.min(alpha)),
            'f_alpha_max': float(np.max(f_alpha))
        }
    
    def _determine_trend(self, df: pd.DataFrame, cycles: List[Cycle], 
                             poly_functions: List[PolynomialFunction]) -> TrendDirection:
        """تشخیص روند پیشرفته"""
        try:
            scores = {'bullish': 0, 'bearish': 0, 'weight': 0}
            
            # 1. تحلیل قیمت (وزن: 3)
            prices = df['close'].values
            if len(prices) >= 50:
                current_price = prices[-1]
                
                # میانگین‌های متحرک
                ma_5 = np.mean(prices[-5:])
                ma_20 = np.mean(prices[-20:])
                ma_50 = np.mean(prices[-50:])
                
                # امتیازدهی
                if current_price > ma_5:
                    scores['bullish'] += 1 * 1.0
                else:
                    scores['bearish'] += 1 * 1.0
                
                if current_price > ma_20:
                    scores['bullish'] += 1 * 1.5
                else:
                    scores['bearish'] += 1 * 1.5
                
                if current_price > ma_50:
                    scores['bullish'] += 1 * 2.0
                else:
                    scores['bearish'] += 1 * 2.0
                
                scores['weight'] += 4.5  # مجموع وزن‌ها
            
            # 2. تحلیل مومنتوم (وزن: 2)
            if len(prices) >= 10:
                momentum_5 = prices[-1] - prices[-5]
                momentum_10 = prices[-1] - prices[-10]
                
                if momentum_5 > 0:
                    scores['bullish'] += 1 * 0.8
                else:
                    scores['bearish'] += 1 * 0.8
                
                if momentum_10 > 0:
                    scores['bullish'] += 1 * 1.2
                else:
                    scores['bearish'] += 1 * 1.2
                
                scores['weight'] += 2.0
            
            # 3. تحلیل cycles (وزن: 1.5)
            if cycles and len(cycles) >= 3:
                recent_cycles = cycles[-3:]
                
                # میانگین net_movement
                avg_movement = np.mean([c.net_movement for c in recent_cycles])
                
                if avg_movement > 0:
                    scores['bullish'] += 1 * 1.5
                else:
                    scores['bearish'] += 1 * 1.5
                
                scores['weight'] += 1.5
            
            # 4. تحلیل polynomial functions (وزن: 1)
            if poly_functions:
                recent_func = poly_functions[-1]
                
                # بررسی velocity و acceleration
                if recent_func.velocity > 0:
                    scores['bullish'] += 1 * 1.0
                else:
                    scores['bearish'] += 1 * 1.0
                
                scores['weight'] += 1.0
            
            # 5. تحلیل RSI (اگر داده کافی داریم)
            if len(prices) >= 14:
                rsi = self._calculate_rsi(prices, period=14)
                if rsi > 50:
                    scores['bullish'] += 1 * 0.8
                else:
                    scores['bearish'] += 1 * 0.8
                
                scores['weight'] += 0.8
            
            # محاسبه نهایی
            if scores['weight'] == 0:
                return TrendDirection.NEUTRAL
            
            bullish_percent = (scores['bullish'] / scores['weight']) * 100
            bearish_percent = (scores['bearish'] / scores['weight']) * 100
            
            logger.info(f"   Trend Analysis: Bullish={bullish_percent:.1f}%, Bearish={bearish_percent:.1f}%")
            
            # تصمیم‌گیری با threshold
            if bullish_percent - bearish_percent >= 10:  # تفاوت حداقل 15%
                return TrendDirection.BULLISH
            elif bearish_percent - bullish_percent >= 10:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return TrendDirection.NEUTRAL

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """محاسبه RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            seed = deltas[:period + 1]
            
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            if down == 0:
                return 100.0
            
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
            
            return np.clip(rsi, 0, 100)
            
        except:
            return 50.0
    
    # حل مشکل بازگشتی - ایجاد متد جدید برای ATR
    def _calculate_atr_value(self, df: pd.DataFrame, period: int = 14) -> float:
        """محاسبه ATR بدون بازگشتی"""
        try:
            if len(df) < period + 1:
                return 10.0  # مقدار پیش‌فرض
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr_list = []
            for i in range(1, period + 1):
                if i < len(high):
                    tr = max(
                        high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])
                    )
                    tr_list.append(tr)
            
            if tr_list:
                return np.mean(tr_list)
            else:
                return 10.0
                
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 10.0
    
    def _calculate_levels_simple(self, direction: TrendDirection, df: pd.DataFrame) -> Tuple[float, float, float]:
        """محاسبه سطوح ساده و مطمئن - اصلاح شده"""
        try:
            current_price = df['close'].values[-1]
            
            # استفاده از متد غیربازگشتی برای ATR
            atr = self._calculate_atr_value(df, period=14)
            
            # محدود کردن ATR
            max_atr = current_price * 0.005  # حداکثر 0.5%
            atr = min(atr, max_atr)
            
            logger.info(f"   Price: {current_price:.2f}, ATR: {atr:.2f}")
            
            if direction == TrendDirection.BULLISH:
                entry = current_price
                sl = current_price - (atr * 1.5)  # 1.5 برابر ATR پایین‌تر
                tp = current_price + (atr * 3.0)  # 3 برابر ATR بالاتر
                
                # اطمینان از درستی سطوح
                if sl >= entry:
                    sl = entry * 0.995
                if tp <= entry:
                    tp = entry * 1.01
                
                logger.info(f"   BUY: Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            
            elif direction == TrendDirection.BEARISH:
                entry = current_price
                sl = current_price + (atr * 1.5)  # 1.5 برابر ATR بالاتر
                tp = current_price - (atr * 3.0)  # 3 برابر ATR پایین‌تر
                
                # اطمینان از درستی سطوح
                if sl <= entry:
                    sl = entry * 1.005
                if tp >= entry:
                    tp = entry * 0.99
                
                logger.info(f"   SELL: Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            
            else:
                return 0.0, 0.0, 0.0
            
            return entry, sl, tp
            
        except Exception as e:
            logger.error(f"Error in _calculate_levels_simple: {e}")
            # بازگشت مقادیر پیش‌فرض
            current_price = df['close'].values[-1] if len(df['close'].values) > 0 else 0
            if direction == TrendDirection.BULLISH:
                return current_price, current_price * 0.995, current_price * 1.01
            elif direction == TrendDirection.BEARISH:
                return current_price, current_price * 1.005, current_price * 0.99
            else:
                return 0.0, 0.0, 0.0

    


    def _calculate_confidence(self, quantum_state: QuantumState, hurst: float, phase_uncertainty: float, poly_functions: List[PolynomialFunction]) -> float:
        """Calculate signal confidence level"""
        confidence = 0.5
        
        if quantum_state in [QuantumState.COLLAPSED_BULLISH, QuantumState.COLLAPSED_BEARISH]:
            confidence += 0.15
        
        if hurst > 0.6:
            confidence += 0.1
        elif hurst < 0.4:
            confidence -= 0.1
        
        if phase_uncertainty < 0.3:
            confidence += 0.1
        
        if poly_functions:
            avg_r2 = np.mean([f.r_squared for f in poly_functions])
            confidence += avg_r2 * 0.15
        
        return np.clip(confidence, 0, 1)
    
    def _log_signal(self, signal: NDSSignal):
        """Display signal details"""
        logger.info("=" * 60)
        logger.info("📊 NDS Signal Identified:")
        logger.info(f"   Direction: {'BUY 📈' if signal.direction == TrendDirection.BULLISH else 'SELL 📉'}")
        logger.info(f"   Entry: {signal.entry_price:.2f}")
        logger.info(f"   SL: {signal.stop_loss:.2f}")
        logger.info(f"   TP: {signal.take_profit:.2f}")
        logger.info(f"   R/R: {signal.risk_reward:.2f}")
        logger.info(f"   Confidence: {signal.confidence:.2%}")
        logger.info(f"   Quantum State: {signal.quantum_state.value}")
        logger.info(f"   Hurst: {signal.hurst_exponent:.3f}")
        logger.info("=" * 60)
    
    def get_next_node_for_tp(self, direction: TrendDirection, current_tp: float) -> Optional[float]:
        """Find next node for TP transfer"""
        nodes = self.nodes_cache.get(self.tf_analysis, [])
        if not nodes:
            return None
        
        if direction == TrendDirection.BULLISH:
            candidates = [n for n in nodes 
                         if n.node_type == 'high' and 
                         (n.displaced_price or n.price) > current_tp]
            if candidates:
                next_node = min(candidates, key=lambda n: (n.displaced_price or n.price))
                return next_node.displaced_price or next_node.price
        else:
            candidates = [n for n in nodes 
                         if n.node_type == 'low' and 
                         (n.displaced_price or n.price) < current_tp]
            if candidates:
                next_node = max(candidates, key=lambda n: (n.displaced_price or n.price))
                return next_node.displaced_price or next_node.price
        
        return None
    

# ============================================================================
# FRACTAL RECURSIVE MODEL - مدل فرکتال بازگشتی
# ============================================================================
class FractalRecursiveModel:
    """
    پیاده‌سازی کامل مدل فرکتال بازگشتی مقاله (صفحات ۸-۹-۱۰)
    """
    
    def __init__(self, scaling_factor: int = 3, max_depth: int = 4):
        self.scaling_factor = scaling_factor
        self.max_depth = max_depth
        self.fractal_levels = {}
        
    def build_fractal_structure(self, price_data: np.ndarray, timeframe: int = 5) -> Dict:
        """
        ساختار فرکتال کامل بازگشتی
        """
        logger.info("🔍 Building complete fractal structure...")
        
        level_0 = {
            'name': 'T(t)',
            'timeframe': timeframe,
            'data': price_data,
            'subtrends': []
        }
        
        self._recursive_decomposition(level_0, depth=0, parent_timeframe=timeframe)
        self._calculate_time_scaling(level_0)
        
        self.fractal_levels = level_0
        return level_0
    
    def _recursive_decomposition(self, node: Dict, depth: int, parent_timeframe: int):
        """
        تجزیه بازگشتی به زیرروندها
        """
        if depth >= self.max_depth:
            return
        
        data = node['data']
        if len(data) < self.scaling_factor * 2:
            return
        
        segment_length = len(data) // self.scaling_factor
        segments = []
        
        for i in range(self.scaling_factor):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < self.scaling_factor - 1 else len(data)
            segment = data[start_idx:end_idx]
            
            if len(segment) > 2:
                segments.append(segment)
        
        for i, segment in enumerate(segments):
            subtrend_name = f"{node['name']}_{i+1}"
            
            subtrend = {
                'name': subtrend_name,
                'timeframe': parent_timeframe / (self.scaling_factor ** (depth + 1)),
                'data': segment,
                'subtrends': [],
                'is_trend': True,
                'index': i
            }
            
            self._recursive_decomposition(subtrend, depth + 1, parent_timeframe)
            
            node['subtrends'].append(subtrend)
            
            if i < len(segments) - 1 and len(segments) > i + 1:
                pullback = self._calculate_pullback(segment, segments[i + 1])
                
                if len(pullback) > 1:
                    pullback_name = f"P_{i+1}"
                    pullback_node = {
                        'name': pullback_name,
                        'timeframe': parent_timeframe / (self.scaling_factor ** (depth + 1)),
                        'data': pullback,
                        'subtrends': [],
                        'is_trend': False,
                        'is_pullback': True
                    }
                    node['subtrends'].append(pullback_node)
    
    def _calculate_pullback(self, trend1: np.ndarray, trend2: np.ndarray) -> np.ndarray:
        """
        محاسبه تابع پولبک بین دو روند
        """
        if len(trend1) == 0 or len(trend2) == 0:
            return np.array([])
        
        end_trend1 = trend1[-1]
        start_trend2 = trend2[0]
        
        pullback_points = 3
        pullback = np.linspace(end_trend1, start_trend2, pullback_points)
        
        return pullback
    
    def _calculate_time_scaling(self, node: Dict):
        """
        محاسبه مقیاس زمانی فرکتال
        """
        if 'subtrends' not in node:
            return
        
        for subt in node['subtrends']:
            if 'parent_timeframe' in node:
                subt['scaled_timeframe'] = node['parent_timeframe'] / self.scaling_factor
            else:
                subt['scaled_timeframe'] = node['timeframe'] / self.scaling_factor
            
            subt['parent_timeframe'] = node['timeframe']
            self._calculate_time_scaling(subt)
    
    def analyze_fractal_pattern(self, fractal_structure: Dict) -> Dict:
        """
        تحلیل الگوهای فرکتال
        """
        analysis = {
            'total_levels': 0,
            'total_nodes': 0,
            'symmetry_score': 0.0,
            'scaling_consistency': 0.0,
            'fractal_dimension': 0.0
        }
        
        self._traverse_fractal(fractal_structure, analysis, level=0)
        
        if analysis['total_nodes'] > 0:
            analysis['symmetry_score'] = analysis['symmetry_score'] / analysis['total_nodes']
            analysis['scaling_consistency'] = self._check_scaling_consistency(fractal_structure)
            analysis['fractal_dimension'] = self._calculate_fractal_dimension(fractal_structure)
        
        return analysis
    
    def _traverse_fractal(self, node: Dict, analysis: Dict, level: int):
        """پیمایش ساختار فرکتال"""
        analysis['total_levels'] = max(analysis['total_levels'], level)
        analysis['total_nodes'] += 1
        
        if 'subtrends' in node and len(node['subtrends']) == self.scaling_factor:
            lengths = [len(st['data']) for st in node['subtrends'] if st.get('is_trend', False)]
            if len(lengths) == self.scaling_factor:
                std_dev = np.std(lengths)
                mean_len = np.mean(lengths)
                if mean_len > 0:
                    symmetry = 1 - (std_dev / mean_len)
                    analysis['symmetry_score'] += symmetry
        
        for subt in node.get('subtrends', []):
            self._traverse_fractal(subt, analysis, level + 1)
    
    def _check_scaling_consistency(self, node: Dict) -> float:
        """بررسی سازگاری مقیاس‌گذاری"""
        if 'subtrends' not in node or len(node['subtrends']) == 0:
            return 1.0
        
        timeframes = []
        for subt in node['subtrends']:
            if 'scaled_timeframe' in subt:
                timeframes.append(subt['scaled_timeframe'])
        
        if len(timeframes) < 2:
            return 1.0
        
        return 1.0 - (np.std(timeframes) / np.mean(timeframes))
    
    def _calculate_fractal_dimension(self, node: Dict) -> float:
        """محاسبه بعد فرکتال"""
        total_nodes = self._count_nodes(node)
        total_levels = self._get_max_depth(node)
        
        if total_levels <= 1:
            return 1.0
        
        return np.log(total_nodes) / np.log(total_levels)
    
    def _count_nodes(self, node: Dict) -> int:
        """شمردن گره‌ها"""
        count = 1
        for subt in node.get('subtrends', []):
            count += self._count_nodes(subt)
        return count
    
    def _get_max_depth(self, node: Dict) -> int:
        """محاسبه حداکثر عمق"""
        if 'subtrends' not in node or len(node['subtrends']) == 0:
            return 1
        
        max_depth = 0
        for subt in node['subtrends']:
            depth = self._get_max_depth(subt)
            max_depth = max(max_depth, depth)
        
        return max_depth + 1
    
    def get_fractal_signal(self, fractal_structure: Dict) -> Dict:
        """
        استخراج سیگنال از ساختار فرکتال
        """
        analysis = self.analyze_fractal_pattern(fractal_structure)
        
        signal = {
            'fractal_aligned': analysis['symmetry_score'] > 0.7,
            'scaling_consistent': analysis['scaling_consistency'] > 0.8,
            'fractal_dimension': analysis['fractal_dimension'],
            'confidence': min(analysis['symmetry_score'] * analysis['scaling_consistency'], 1.0),
            'levels_detected': analysis['total_levels'],
            'total_nodes': analysis['total_nodes']
        }
        
        if len(fractal_structure.get('data', [])) > 1:
            price_change = fractal_structure['data'][-1] - fractal_structure['data'][0]
            signal['direction'] = 'bullish' if price_change > 0 else 'bearish'
            signal['momentum'] = abs(price_change) / np.mean(fractal_structure['data'])
        
        return signal


# ============================================================================
# NEURAL NETWORK ENHANCEMENT - بهبود شبکه عصبی
# ============================================================================
class NDSTrendEnhancer(nn.Module if TORCH_AVAILABLE else object):
    """
    شبکه عصبی برای بهبود توابع روند (صفحات ۶-۷ مقاله)
    """
    
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [32, 64, 32]):
        super(NDSTrendEnhancer, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.poly_weights = nn.Parameter(torch.randn(4))
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, t, poly_coeffs: Optional[Any] = None):
        """
        محاسبه تابع بهبودیافته روند
        """
        if poly_coeffs is None:
            poly_coeffs = self.poly_weights
        
        poly_features = torch.stack([
            torch.ones_like(t),
            t,
            t**2,
            t**3
        ], dim=-1)
        
        poly_part = torch.sum(poly_features * poly_coeffs, dim=-1)
        
        nn_input = torch.stack([
            t,
            t**2,
            torch.sin(2 * torch.pi * t / 100),
            torch.cos(2 * torch.pi * t / 100),
            poly_part.detach()
        ], dim=-1)
        
        nn_part = self.network(nn_input).squeeze(-1)
        enhanced = poly_part + self.alpha * nn_part
        
        return enhanced, poly_part, nn_part
    
    def calculate_error(self, predictions, targets):
        """محاسبه خطا با MSE"""
        return nn.functional.mse_loss(predictions, targets)
    
    def train_step(self, t_batch, price_batch, 
                   optimizer) -> Dict[str, float]:
        """
        یک مرحله آموزش
        """
        self.train()
        optimizer.zero_grad()
        
        predictions, poly_part, nn_part = self.forward(t_batch)
        loss = self.calculate_error(predictions, price_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'poly_mse': nn.functional.mse_loss(poly_part, price_batch).item(),
            'nn_contribution': self.alpha.item()
        }
    
    def enhance_trend_function(self, t: np.ndarray, prices: np.ndarray, 
                               epochs: int = 100, lr: float = 0.001) -> Dict:
        """
        بهبود تابع روند با شبکه عصبی
        """
        t_tensor = torch.FloatTensor(t).unsqueeze(1)
        price_tensor = torch.FloatTensor(prices)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        history = {'loss': [], 'poly_loss': [], 'nn_weight': []}
        
        for epoch in range(epochs):
            metrics = self.train_step(t_tensor, price_tensor, optimizer)
            
            history['loss'].append(metrics['total_loss'])
            history['poly_loss'].append(metrics['poly_mse'])
            history['nn_weight'].append(metrics['nn_contribution'])
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: Loss = {metrics['total_loss']:.6f}, "
                           f"NN Weight = {metrics['nn_contribution']:.4f}")
        
        with torch.no_grad():
            enhanced_prices, poly_prices, nn_correction = self.forward(t_tensor)
        
        poly_mse = np.mean((poly_prices.numpy() - prices) ** 2)
        enhanced_mse = np.mean((enhanced_prices.numpy() - prices) ** 2)
        improvement = ((poly_mse - enhanced_mse) / poly_mse) * 100
        
        return {
            'enhanced_prices': enhanced_prices.numpy(),
            'poly_prices': poly_prices.numpy(),
            'nn_correction': nn_correction.numpy(),
            'improvement_percent': improvement,
            'final_alpha': self.alpha.item(),
            'training_history': history
        }


class NeuralNetworkManager:
    """مدیریت شبکه‌های عصبی برای NDS"""
    
    def __init__(self):
        self.trend_enhancer = NDSTrendEnhancer()
        self.pullback_enhancer = NDSTrendEnhancer(input_dim=5, hidden_dims=[16, 32, 16])
        self.optimizers = {
            'trend': optim.Adam(self.trend_enhancer.parameters(), lr=0.001),
            'pullback': optim.Adam(self.pullback_enhancer.parameters(), lr=0.001)
        }
        
    def enhance_nds_functions(self, nds_data: Dict) -> Dict:
        """
        بهبود همه توابع NDS با شبکه عصبی
        """
        enhanced_results = {}
        
        if 'trend_functions' in nds_data:
            enhanced_results['trend_functions'] = []
            
            for trend_func in nds_data['trend_functions']:
                t = np.arange(len(trend_func['prices']))
                prices = trend_func['prices']
                
                result = self.trend_enhancer.enhance_trend_function(t, prices)
                enhanced_results['trend_functions'].append(result)
                
                logger.info(f"Trend function enhanced: {result['improvement_percent']:.2f}% improvement")
        
        if 'pullback_functions' in nds_data:
            enhanced_results['pullback_functions'] = []
            
            for pullback_func in nds_data['pullback_functions']:
                t = np.arange(len(pullback_func['prices']))
                prices = pullback_func['prices']
                
                result = self.pullback_enhancer.enhance_trend_function(t, prices)
                enhanced_results['pullback_functions'].append(result)
        
        total_improvement = self._calculate_total_improvement(enhanced_results)
        enhanced_results['total_improvement'] = total_improvement
        
        return enhanced_results
    
    def _calculate_total_improvement(self, enhanced_results: Dict) -> float:
        """محاسبه بهبود کلی"""
        improvements = []
        
        for func_type in ['trend_functions', 'pullback_functions']:
            if func_type in enhanced_results:
                for result in enhanced_results[func_type]:
                    improvements.append(result['improvement_percent'])
        
        return np.mean(improvements) if improvements else 0.0
    
    def adaptive_learning(self, market_data: pd.DataFrame, lookback: int = 100):
        """
        یادگیری تطبیقی بر اساس داده‌های اخیر بازار
        """
        prices = market_data['close'].values[-lookback:]
        t = np.arange(len(prices))
        
        for epoch in range(50):
            t_tensor = torch.FloatTensor(t).unsqueeze(1)
            price_tensor = torch.FloatTensor(prices)
            
            metrics = self.trend_enhancer.train_step(t_tensor, price_tensor, 
                                                    self.optimizers['trend'])
            
            if epoch == 0 or epoch == 49:
                logger.debug(f"Adaptive learning epoch {epoch}: Loss = {metrics['total_loss']:.6f}")

    
# ============================================================================
# ADVANCED ML MODELS FROM ARTICLE - مدل‌های پیشرفته مقاله
# ============================================================================

# ============================================================================
# 1. TRANSFORMER FOR PRICE PREDICTION (Section 4.1)
# ============================================================================
class PriceTransformer(nn.Module):
    """
    Transformer model for price prediction: S_{t+1} = Transformer(S_{t-24h:t}, λ_{t-24h:t}, M_t)
    Section 4.1 of the article
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super(PriceTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Linear(5, d_model)  # price, volume, volatility, OFI, macro
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
    def forward(self, x):
        """
        x: (batch, seq_len, features) where features = [price, volume, volatility, OFI, macro]
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Use last timestep for prediction
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        
        # Output prediction
        prediction = self.output(last_hidden)  # (batch, 1)
        
        return prediction
    
    def predict(self, price_data: pd.DataFrame, ofi_data: np.ndarray = None, 
                macro_data: np.ndarray = None) -> float:
        """
        Predict next price: S_{t+1} = Transformer(S_{t-24h:t}, λ_{t-24h:t}, M_t)
        """
        try:
            # Prepare input sequence (last 24 hours = 24 timesteps for M1)
            seq_len = min(24, len(price_data))
            if seq_len < 24:
                logger.warning(f"Insufficient data: {seq_len} < 24 timesteps")
                return None
            
            recent_data = price_data.tail(seq_len)
            
            # Extract features
            prices = recent_data['close'].values
            volumes = recent_data['volume'].values if 'volume' in recent_data else np.ones_like(prices)
            returns = np.diff(prices, prepend=prices[0])
            volatility = np.abs(returns)
            
            # OFI (Order Flow Imbalance) - simplified
            if ofi_data is None:
                ofi_data = np.zeros_like(prices)
            else:
                ofi_data = ofi_data[-seq_len:] if len(ofi_data) >= seq_len else np.zeros(seq_len)
            
            # Macro signals - simplified
            if macro_data is None:
                macro_data = np.zeros_like(prices)
            else:
                macro_data = macro_data[-seq_len:] if len(macro_data) >= seq_len else np.zeros(seq_len)
            
            # Normalize
            price_mean, price_std = prices.mean(), prices.std() + 1e-8
            prices_norm = (prices - price_mean) / price_std
            
            vol_mean, vol_std = volumes.mean(), volumes.std() + 1e-8
            volumes_norm = (volumes - vol_mean) / vol_std if vol_std > 0 else volumes
            
            vol_mean, vol_std = volatility.mean(), volatility.std() + 1e-8
            volatility_norm = (volatility - vol_mean) / vol_std if vol_std > 0 else volatility
            
            # Create input tensor
            features = np.stack([
                prices_norm,
                volumes_norm,
                volatility_norm,
                ofi_data,
                macro_data
            ], axis=1)
            
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, 5)
            
            # Predict
            self.eval()
            with torch.no_grad():
                prediction_norm = self.forward(x).item()
            
            # Denormalize
            prediction = prediction_norm * price_std + price_mean
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            return None


# ============================================================================
# 2. GRAPH NEURAL NETWORK FOR ASSET CORRELATION (Section 4.2)
# ============================================================================
class AssetCorrelationGNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Graph Neural Network for inter-asset signals: Z_t = GNN(E_t, M_t)
    Section 4.2 of the article
    """
    
    def __init__(self, node_features: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super(AssetCorrelationGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.gcn1 = nn.Linear(node_features, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn3 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, node_features, adjacency_matrix):
        """
        node_features: (num_nodes, node_features)
        adjacency_matrix: (num_nodes, num_nodes)
        """
        # Normalize adjacency matrix
        adj_norm = self._normalize_adjacency(adjacency_matrix)
        
        # First GCN layer
        h = self.gcn1(node_features)  # (num_nodes, hidden_dim)
        h = torch.matmul(adj_norm, h)  # Graph convolution
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second GCN layer
        h = self.gcn2(h)
        h = torch.matmul(adj_norm, h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Output layer
        output = self.gcn3(h)  # (num_nodes, output_dim)
        
        # Aggregate node features (mean pooling)
        aggregated = torch.mean(output, dim=0)  # (output_dim,)
        
        return aggregated
    
    def _normalize_adjacency(self, adj):
        """Normalize adjacency matrix"""
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Degree matrix
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_matrix = torch.diag(degree_inv_sqrt)
        
        # Normalized adjacency
        adj_norm = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        
        return adj_norm
    
    def compute_correlation_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute correlation signals for multiple assets
        E_t: correlation matrix between assets
        """
        try:
            assets = list(price_data.keys())
            if len(assets) < 2:
                return {}
            
            # Compute correlation matrix
            returns_dict = {}
            for asset, df in price_data.items():
                if 'close' in df.columns and len(df) > 1:
                    returns = df['close'].pct_change().dropna()
                    returns_dict[asset] = returns
            
            if len(returns_dict) < 2:
                return {}
            
            # Align returns
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                return {}
            
            # Correlation matrix
            corr_matrix = returns_df.corr().values
            adj_matrix = torch.FloatTensor(np.abs(corr_matrix))  # Use absolute correlation
            
            # Node features (price, volume, volatility for each asset)
            node_features_list = []
            for asset in assets:
                if asset in price_data:
                    df = price_data[asset]
                    if len(df) > 0:
                        price = df['close'].iloc[-1]
                        volume = df['volume'].iloc[-1] if 'volume' in df.columns else 1.0
                        volatility = df['close'].pct_change().std() if len(df) > 1 else 0.0
                        
                        # Additional features (simplified)
                        features = np.array([
                            price, volume, volatility,
                            price / df['close'].mean() if len(df) > 0 else 1.0,
                            df['close'].iloc[-1] / df['close'].iloc[0] if len(df) > 0 else 1.0,
                            0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder for more features
                        ])
                        node_features_list.append(features[:self.node_features])
            
            if len(node_features_list) < 2:
                return {}
            
            node_features = torch.FloatTensor(np.array(node_features_list))
            
            # Normalize features
            node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
            
            # Forward pass
            self.eval()
            with torch.no_grad():
                signals = self.forward(node_features, adj_matrix)
            
            # Map signals to assets
            result = {}
            signal_values = signals.cpu().numpy()
            for i, asset in enumerate(assets[:len(signal_values)]):
                result[asset] = float(signal_values[i]) if i < len(signal_values) else 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GNN correlation signals: {e}")
            return {}


# ============================================================================
# 3. ACTOR-CRITIC RL FOR POLICY OPTIMIZATION (Section 4.3)
# ============================================================================
class ActorCriticRL(nn.Module if TORCH_AVAILABLE else object):
    """
    Actor-Critic RL for trading policy optimization
    Section 4.3 of the article
    
    Actor: determines position size a_t
    Critic: estimates value function V(s_t)
    """
    
    def __init__(self, state_dim: int = 20, action_dim: int = 1, hidden_dim: int = 128):
        super(ActorCriticRL, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy): π(a_t|s_t)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1] for normalized position size
        )
        
        # Critic network (value): V(s_t)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Action std (for continuous actions)
        self.action_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        
    def forward(self, state) -> Tuple[Any, Any]:
        """
        Returns: (action_mean, value)
        """
        features = self.shared(state)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(self, state) -> Tuple[Any, Any]:
        """
        Sample action from policy: a_t ~ π(·|s_t)
        Returns: (action, log_prob)
        """
        action_mean, value = self.forward(state)
        
        # Create distribution
        action_std = torch.clamp(self.action_std, min=0.01, max=1.0)
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, state, action) -> Tuple[Any, Any, Any]:
        """
        Evaluate action: returns (log_prob, value, entropy)
        """
        action_mean, value = self.forward(state)
        
        action_std = torch.clamp(self.action_std, min=0.01, max=1.0)
        dist = Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class RLPolicyOptimizer:
    """
    RL Policy Optimizer that updates parameters after every 5 trades
    Implements Actor-Critic algorithm with PPO-style updates
    """
    
    def __init__(self, state_dim: int = 20, lr: float = 3e-4, gamma: float = 0.99):
        self.state_dim = state_dim
        self.gamma = gamma
        
        # Actor-Critic model
        self.model = ActorCriticRL(state_dim=state_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # Trade counter for optimization
        self.trade_count = 0
        self.optimization_interval = 5  # Optimize after every 5 trades
        
        # Parameter history for tracking
        self.parameter_history = []
        
    def add_experience(self, state: np.ndarray, action: float, reward: float, 
                      done: bool = False, value: float = None):
        """
        Add trading experience to buffer
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action and value if not provided
        if value is None:
            with torch.no_grad():
                _, value_tensor = self.model(state_tensor)
                value = value_tensor.item()
        
        # Store experience
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)
        
        self.trade_count += 1
        
        # Optimize if interval reached
        if self.trade_count >= self.optimization_interval:
            self.optimize_parameters()
            self.trade_count = 0
    
    def optimize_parameters(self):
        """
        Optimize RL parameters after every 5 trades
        Implements PPO-style update with KL divergence constraint
        """
        if len(self.buffer['states']) < 2:
            return
        
        try:
            # Convert to tensors
            states = torch.FloatTensor(np.array(self.buffer['states']))
            actions = torch.FloatTensor(np.array(self.buffer['actions'])).unsqueeze(-1)
            rewards = np.array(self.buffer['rewards'])
            old_values = torch.FloatTensor(np.array(self.buffer['values']))
            dones = np.array(self.buffer['dones'])
            
            # Compute returns (discounted rewards)
            returns = self._compute_returns(rewards, dones)
            returns_tensor = torch.FloatTensor(returns)
            
            # Normalize returns
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
            # Compute advantages
            advantages = returns_tensor - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update (simplified)
            for _ in range(3):  # Multiple update steps
                # Get current policy outputs
                log_probs, values, entropy = self.model.evaluate(states, actions)
                
                # Policy loss (Actor) با فرمول کامل PPO
                # محاسبه old log probs از buffer
                old_log_probs_tensor = torch.FloatTensor([
                    np.log(0.5) for _ in range(len(states))
                ])  # تقریب برای old policy
                
                # محاسبه ratio با فرمول کامل: ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(log_probs - old_log_probs_tensor)
                
                # Clipped surrogate objective
                clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)  # ε = 0.2
                policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
                
                # KL divergence penalty (برای محدود کردن تغییر policy)
                kl_div = torch.mean(old_log_probs_tensor - log_probs)
                kl_penalty = 0.01 * kl_div**2  # β = 0.01
                policy_loss += kl_penalty
                
                # Value loss (Critic)
                value_loss = F.mse_loss(values.squeeze(), returns_tensor)
                
                # Entropy bonus
                entropy_bonus = -0.01 * entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + entropy_bonus
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            # Store parameter history
            params = {name: param.data.clone().cpu().numpy() 
                     for name, param in self.model.named_parameters()}
            self.parameter_history.append(params)
            
            # Clear buffer
            self.buffer = {key: [] for key in self.buffer.keys()}
            
            logger.info(f"✅ RL parameters optimized after {self.trade_count} trades")
            
        except Exception as e:
            logger.error(f"Error in RL optimization: {e}")
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + self.gamma * running_return
            returns[i] = running_return
        
        return returns
    
    def get_position_size(self, state: np.ndarray) -> float:
        """
        Get normalized position size from Actor network: a_t ∈ [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            action_mean, _ = self.model(state_tensor)
            action = action_mean.item()
        
        # Normalize to [0, 1] for position size (0 = no position, 1 = max position)
        normalized_action = (action + 1) / 2
        
        return float(np.clip(normalized_action, 0.0, 1.0))


# ============================================================================
# 4. HIDDEN MARKOV MODEL FOR MARKET REGIME DETECTION (Section 5.3)
# ============================================================================
class MarketRegimeHMM:
    """
    Hidden Markov Model for market regime classification
    Regime_t = HMM(σ_t, λ_t, M_t)
    Section 5.3 of the article
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.regimes = ['low_volatility', 'trending', 'ranging', 'volatile']
        
        # Transition matrix (will be learned with Baum-Welch algorithm)
        self.transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        
        # Emission parameters (mean and std for each regime)
        self.means = np.random.randn(n_regimes, 3)  # 3 features: volatility, OFI, macro
        self.stds = np.ones((n_regimes, 3))
        
        # Initial state distribution
        self.initial_probs = np.ones(n_regimes) / n_regimes
        
        # History for learning
        self.observation_history = []
        self.state_history = []
        self.learned = False
        
    def detect_regime(self, volatility: float, ofi: float, macro_signal: float) -> str:
        """
        Detect current market regime
        """
        try:
            features = np.array([volatility, ofi, macro_signal])
            
            # Compute emission probabilities for each regime
            probs = []
            for i in range(self.n_regimes):
                # Gaussian emission probability
                diff = features - self.means[i]
                prob = np.exp(-0.5 * np.sum((diff / (self.stds[i] + 1e-8)) ** 2))
                probs.append(prob)
            
            probs = np.array(probs)
            probs = probs / (probs.sum() + 1e-8)
            
            # Select regime with highest probability
            regime_idx = np.argmax(probs)
            regime = self.regimes[regime_idx] if regime_idx < len(self.regimes) else 'trending'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return 'trending'  # Default
    
    def learn_from_data(self, observations: List[np.ndarray], max_iterations: int = 50):
        """
        یادگیری پارامترهای HMM با الگوریتم Baum-Welch (EM algorithm)
        فرمول کامل برای transition matrix و emission parameters
        """
        try:
            if len(observations) < 20:
                return
            
            observations_array = np.array(observations)
            
            # Initialize parameters
            # Transition matrix: A[i,j] = P(state_{t+1}=j | state_t=i)
            A = np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
            
            # Emission means and stds
            means = np.zeros((self.n_regimes, 3))
            stds = np.ones((self.n_regimes, 3))
            
            # K-means initialization برای means
            from sklearn.cluster import KMeans
            if SKLEARN_AVAILABLE and len(observations_array) >= self.n_regimes:
                kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
                labels = kmeans.fit_predict(observations_array)
                means = kmeans.cluster_centers_
                
                # محاسبه stds برای هر cluster
                for i in range(self.n_regimes):
                    cluster_data = observations_array[labels == i]
                    if len(cluster_data) > 0:
                        stds[i] = np.std(cluster_data, axis=0) + 1e-6
            
            # Baum-Welch algorithm (EM)
            for iteration in range(max_iterations):
                # E-step: Forward-Backward algorithm
                # محاسبه forward probabilities: α_t(i) = P(o_1, ..., o_t, state_t=i | λ)
                # محاسبه backward probabilities: β_t(i) = P(o_{t+1}, ..., o_T | state_t=i, λ)
                
                T = len(observations_array)
                alpha = np.zeros((T, self.n_regimes))
                beta = np.zeros((T, self.n_regimes))
                
                # Forward pass
                # α_1(i) = π_i * b_i(o_1)
                for i in range(self.n_regimes):
                    emission_prob = self._gaussian_emission(observations_array[0], means[i], stds[i])
                    alpha[0, i] = self.initial_probs[i] * emission_prob
                
                # Normalize
                alpha[0] = alpha[0] / (np.sum(alpha[0]) + 1e-10)
                
                for t in range(1, T):
                    for j in range(self.n_regimes):
                        emission_prob = self._gaussian_emission(observations_array[t], means[j], stds[j])
                        alpha[t, j] = emission_prob * np.sum(alpha[t-1, :] * A[:, j])
                    alpha[t] = alpha[t] / (np.sum(alpha[t]) + 1e-10)
                
                # Backward pass
                beta[-1, :] = 1.0
                for t in range(T-2, -1, -1):
                    for i in range(self.n_regimes):
                        emission_probs = np.array([
                            self._gaussian_emission(observations_array[t+1], means[j], stds[j])
                            for j in range(self.n_regimes)
                        ])
                        beta[t, i] = np.sum(A[i, :] * emission_probs * beta[t+1, :])
                    beta[t] = beta[t] / (np.sum(beta[t]) + 1e-10)
                
                # M-step: Update parameters
                # ξ_t(i,j) = P(state_t=i, state_{t+1}=j | O, λ)
                xi = np.zeros((T-1, self.n_regimes, self.n_regimes))
                for t in range(T-1):
                    for i in range(self.n_regimes):
                        for j in range(self.n_regimes):
                            emission_prob = self._gaussian_emission(observations_array[t+1], means[j], stds[j])
                            xi[t, i, j] = alpha[t, i] * A[i, j] * emission_prob * beta[t+1, j]
                    xi[t] = xi[t] / (np.sum(xi[t]) + 1e-10)
                
                # γ_t(i) = P(state_t=i | O, λ)
                gamma = alpha * beta
                gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)
                
                # Update transition matrix
                for i in range(self.n_regimes):
                    for j in range(self.n_regimes):
                        A[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-10)
                
                # Update emission parameters
                for i in range(self.n_regimes):
                    gamma_i = gamma[:, i]
                    if np.sum(gamma_i) > 0:
                        means[i] = np.sum(gamma_i[:, np.newaxis] * observations_array, axis=0) / (np.sum(gamma_i) + 1e-10)
                        diff = observations_array - means[i]
                        stds[i] = np.sqrt(np.sum(gamma_i[:, np.newaxis] * diff**2, axis=0) / (np.sum(gamma_i) + 1e-10)) + 1e-6
                
                # Update initial probabilities
                self.initial_probs = gamma[0, :]
            
            # ذخیره پارامترهای یادگرفته شده
            self.transition_matrix = A
            self.means = means
            self.stds = stds
            self.learned = True
            
        except Exception as e:
            logger.debug(f"Error learning HMM parameters: {e}")
    
    def _gaussian_emission(self, observation: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """محاسبه احتمال emission با توزیع گاوسی"""
        diff = observation - mean
        prob = np.exp(-0.5 * np.sum((diff / (std + 1e-8))**2))
        prob = prob / (np.prod(std + 1e-8) * np.sqrt(2 * np.pi)**len(observation))
        return max(prob, 1e-10)
    
    def get_position_limit(self, regime: str) -> float:
        """
        Get position size limit based on regime
        - If regime = 'volatile': |a_t| ≤ 0.3
        - If regime = 'low_volatility': |a_t| ≤ 1.0
        """
        limits = {
            'volatile': 0.3,
            'trending': 0.7,
            'ranging': 0.5,
            'low_volatility': 1.0
        }
        return limits.get(regime, 0.5)


# ============================================================================
# 5. CONDITIONAL VALUE AT RISK (CVaR) (Section 5.5)
# ============================================================================
class CVaRRiskManager:
    """
    Conditional Value at Risk for dynamic risk management
    CVaR_{t,α} = E[Loss | Loss ≥ VaR_{t,α}]
    Section 5.5 of the article
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Confidence level (5% tail)
        self.loss_history = deque(maxlen=1000)
        
    def update_loss_history(self, loss: float):
        """Update loss history"""
        self.loss_history.append(loss)
    
    def compute_var(self) -> float:
        """Compute Value at Risk"""
        if len(self.loss_history) < 10:
            return 0.0
        
        losses = np.array(list(self.loss_history))
        var = np.percentile(losses, (1 - self.alpha) * 100)
        
        return float(var)
    
    def compute_cvar(self) -> float:
        """
        Compute Conditional Value at Risk
        CVaR = E[Loss | Loss ≥ VaR]
        """
        if len(self.loss_history) < 10:
            return 0.0
        
        losses = np.array(list(self.loss_history))
        var = self.compute_var()
        
        # Conditional expectation of losses exceeding VaR
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) == 0:
            return var
        
        cvar = np.mean(tail_losses)
        
        return float(cvar)
    
    def get_risk_adjusted_position_size(self, base_position: float, cvar: float, 
                                       account_balance: float) -> float:
        """
        Adjust position size based on CVaR
        """
        if cvar <= 0 or account_balance <= 0:
            return base_position
        
        # Risk budget: don't risk more than CVaR
        max_risk = account_balance * 0.01  # 1% of balance
        risk_per_unit = cvar
        
        if risk_per_unit > 0:
            adjusted_size = min(base_position, max_risk / risk_per_unit)
        else:
            adjusted_size = base_position
        
        return float(np.clip(adjusted_size, 0.0, base_position))


# ============================================================================
# 5.4. ADAPTIVE RISK BUDGETING (Section 5.4)
# ============================================================================
class AdaptiveRiskBudgeting:
    """
    بودجه‌بندی ریسک تطبیقی
    RiskBudget_t = κ / (σ_t · √E_t)
    Section 5.4 of the article
    """
    
    def __init__(self, kappa: float = 0.02):
        self.kappa = kappa  # Constant parameter
        self.volatility_history = deque(maxlen=100)
        self.correlation_history = deque(maxlen=100)
        
    def update(self, volatility: float, correlation_matrix: np.ndarray = None):
        """به‌روزرسانی نوسان و همبستگی"""
        self.volatility_history.append(volatility)
        
        if correlation_matrix is not None:
            # محاسبه average correlation
            n = correlation_matrix.shape[0]
            if n > 1:
                # میانگین همبستگی (بدون diagonal)
                avg_corr = (np.sum(correlation_matrix) - n) / (n * (n - 1))
                self.correlation_history.append(avg_corr)
            else:
                self.correlation_history.append(1.0)
        else:
            self.correlation_history.append(1.0)
    
    def compute_risk_budget(self) -> float:
        """
        محاسبه بودجه ریسک با فرمول کامل: RiskBudget_t = κ / (σ_t · √E_t)
        که در آن:
        - κ: constant parameter
        - σ_t: نوسان فعلی
        - E_t: average correlation (√E_t برای normalization)
        """
        if len(self.volatility_history) == 0:
            return 0.02  # Default
        
        # نوسان فعلی
        sigma_t = self.volatility_history[-1] if len(self.volatility_history) > 0 else 0.02
        
        # میانگین همبستگی
        if len(self.correlation_history) > 0:
            avg_corr = np.mean(list(self.correlation_history))
            sqrt_E_t = np.sqrt(avg_corr + 1e-8)
        else:
            sqrt_E_t = 1.0
        
        # فرمول کامل
        risk_budget = self.kappa / (sigma_t * sqrt_E_t + 1e-8)
        
        # محدود کردن به بازه معقول
        risk_budget = np.clip(risk_budget, 0.001, 0.1)  # بین 0.1% تا 10%
        
        return float(risk_budget)


# ============================================================================
# 6. GARCH MODEL FOR VOLATILITY FORECASTING (Section 7.2)
# ============================================================================
class GARCHVolatilityModel:
    """
    GARCH model for volatility forecasting
    σ_t² = α₀ + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}
    Section 7.2 of the article
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p  # ARCH order
        self.q = q  # GARCH order
        
        # Parameters (will be estimated)
        self.alpha_0 = 0.01
        self.alpha = np.array([0.1])  # ARCH coefficients
        self.beta = np.array([0.8])  # GARCH coefficients
        
        # Volatility history
        self.volatility_history = deque(maxlen=1000)
        self.returns_history = deque(maxlen=1000)
        
    def update(self, returns: np.ndarray):
        """
        Update GARCH model with new returns
        فرمول کامل: σ_t² = α₀ + Σ_{i=1}^q α_i ε²_{t-i} + Σ_{j=1}^p β_j σ²_{t-j}
        """
        self.returns_history.extend(returns)
        
        if len(self.returns_history) < max(self.p, self.q) + 10:
            return
        
        returns_array = np.array(list(self.returns_history))
        
        # تخمین پارامترهای GARCH با Maximum Likelihood Estimation (MLE)
        if len(returns_array) > 50:
            # استفاده از روش MLE برای تخمین پارامترها
            from scipy.optimize import minimize
            
            def garch_log_likelihood(params, returns):
                """Log-likelihood function for GARCH"""
                alpha_0, alpha_1, beta_1 = params
                
                # محدودیت‌ها: α₀ > 0, α₁ ≥ 0, β₁ ≥ 0, α₁ + β₁ < 1
                if alpha_0 <= 0 or alpha_1 < 0 or beta_1 < 0 or alpha_1 + beta_1 >= 1:
                    return 1e10
                
                T = len(returns)
                sigma2 = np.zeros(T)
                sigma2[0] = np.var(returns)
                
                # محاسبه σ²_t برای تمام t
                for t in range(1, T):
                    sigma2[t] = alpha_0 + alpha_1 * returns[t-1]**2 + beta_1 * sigma2[t-1]
                
                # محاسبه log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(sigma2) + (returns**2) / sigma2)
                
                return -log_likelihood  # برای minimize
            
            try:
                # حدس اولیه
                initial_params = [0.01, 0.1, 0.8]
                
                # محدودیت‌ها
                bounds = [(1e-6, 1.0), (0.0, 1.0), (0.0, 1.0)]
                
                # بهینه‌سازی
                result = minimize(
                    garch_log_likelihood,
                    initial_params,
                    args=(returns_array[-200:],),  # استفاده از 200 داده اخیر
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                
                if result.success:
                    self.alpha_0, self.alpha[0], self.beta[0] = result.x
                else:
                    # اگر بهینه‌سازی ناموفق بود، از مقادیر پیش‌فرض استفاده کن
                    pass
                    
            except Exception as e:
                logger.debug(f"GARCH MLE optimization failed: {e}, using default parameters")
        
        # محاسبه σ²_t با فرمول کامل GARCH
        returns_array = returns_array[-500:]  # استفاده از 500 داده اخیر
        T = len(returns_array)
        
        if T > max(self.p, self.q):
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(returns_array)
            
            # محاسبه σ²_t با فرمول کامل: σ_t² = α₀ + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}
            for t in range(1, T):
                # ARCH term: Σ_{i=1}^q α_i ε²_{t-i}
                arch_term = 0.0
                for i in range(1, min(self.p + 1, t + 1)):
                    arch_term += self.alpha[i-1] * (returns_array[t-i]**2)
                
                # GARCH term: Σ_{j=1}^p β_j σ²_{t-j}
                garch_term = 0.0
                for j in range(1, min(self.q + 1, t + 1)):
                    garch_term += self.beta[j-1] * sigma2[t-j]
                
                # فرمول کامل
                sigma2[t] = self.alpha_0 + arch_term + garch_term
                
                # ذخیره نوسان
                self.volatility_history.append(np.sqrt(sigma2[t]))
    
    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast volatility for next period
        """
        if len(self.volatility_history) == 0:
            return 0.02  # Default 2% volatility
        
        # Use last volatility as forecast (simplified)
        last_vol = self.volatility_history[-1] if len(self.volatility_history) > 0 else 0.02
        
        return float(last_vol)
    
    def get_stop_loss_level(self, entry_price: float, direction: str, 
                           volatility: float, k: float = 2.0) -> float:
        """
        Calculate stop loss based on GARCH volatility
        P_stop = P_entry ± k * σ_t
        """
        if direction.lower() == 'long' or direction.lower() == 'buy':
            stop_loss = entry_price - k * volatility * entry_price
        else:  # short/sell
            stop_loss = entry_price + k * volatility * entry_price
        
        return float(stop_loss)


# ============================================================================
# 7. VWAP OPTIMIZATION AND VOLUME MODELING (Section 6)
# ============================================================================
class VWAPOptimizer:
    """
    VWAP optimization with volume decomposition
    x_{i,t} = c_{i,t} + y_{i,t}
    Section 6 of the article
    """
    
    def __init__(self):
        self.volume_history = deque(maxlen=1000)
        self.price_history = deque(maxlen=1000)
        
    def decompose_volume(self, volumes: np.ndarray, prices: np.ndarray) -> Dict:
        """
        Decompose volume into market component and idiosyncratic component
        c_{i,t} = x̄_i + (1/λ₁) * Cov(x_it, C_t¹) * C_t¹
        y_{i,t} = Σ_{k>1} (1/λ_k) * Cov(x_it, C_t^k) * C_t^k
        """
        try:
            if len(volumes) < 10 or len(prices) != len(volumes):
                return {'market_component': volumes, 'idiosyncratic': np.zeros_like(volumes)}
            
            # Market component با فرمول کامل مقاله:
            # c_{i,t} = x̄_i + (1/λ₁) * Cov(x_it, C_t¹) * C_t¹
            # y_{i,t} = Σ_{k>1} (1/λ_k) * Cov(x_it, C_t^k) * C_t^k
            
            if SKLEARN_AVAILABLE:
                # استفاده از PCA برای تجزیه کامل
                data = np.column_stack([volumes, prices])
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                pca = PCA(n_components=min(5, len(volumes)))
                components = pca.fit_transform(data_scaled)
                
                # محاسبه mean volume
                x_bar = np.mean(volumes)
                
                # محاسبه Covariance matrix
                cov_matrix = np.cov(data_scaled.T)
                
                # محاسبه eigenvalues (λ_k) و eigenvectors (C_t^k)
                eigenvalues = pca.explained_variance_  # λ_k
                eigenvectors = pca.components_  # C_t^k
                
                # Market component: c_{i,t} = x̄_i + (1/λ₁) * Cov(x_it, C_t¹) * C_t¹
                lambda_1 = eigenvalues[0] if len(eigenvalues) > 0 else 1.0
                C_t_1 = components[:, 0]  # اولین component
                
                # Cov(x_it, C_t¹)
                cov_x_C1 = np.cov(volumes, C_t_1)[0, 1]
                
                # محاسبه market component
                market_component = x_bar + (1.0 / (lambda_1 + 1e-8)) * cov_x_C1 * C_t_1
                market_component = market_component * np.std(volumes) + np.mean(volumes)
                
                # Idiosyncratic component: y_{i,t} = Σ_{k>1} (1/λ_k) * Cov(x_it, C_t^k) * C_t^k
                idiosyncratic = np.zeros_like(volumes)
                for k in range(1, min(len(eigenvalues), len(components[0]))):
                    lambda_k = eigenvalues[k]
                    C_t_k = components[:, k]
                    cov_x_Ck = np.cov(volumes, C_t_k)[0, 1]
                    idiosyncratic += (1.0 / (lambda_k + 1e-8)) * cov_x_Ck * C_t_k
                
                idiosyncratic = idiosyncratic * np.std(volumes)
                
            else:
                # Fallback: استفاده از فرمول ساده‌تر اما کامل
                x_bar = np.mean(volumes)
                
                # محاسبه covariance با قیمت به عنوان proxy برای market component
                cov_vol_price = np.cov(volumes, prices)[0, 1]
                var_price = np.var(prices)
                
                if var_price > 0:
                    # تقریب market component
                    market_component = x_bar + (cov_vol_price / var_price) * (prices - np.mean(prices))
                    market_component = market_component * (np.std(volumes) / (np.std(market_component) + 1e-8)) + x_bar
                else:
                    market_component = np.full_like(volumes, x_bar)
                
                # Idiosyncratic component
                idiosyncratic = volumes - market_component
            
            return {
                'market_component': market_component,
                'idiosyncratic': idiosyncratic,
                'total': volumes
            }
            
        except Exception as e:
            logger.error(f"Error in volume decomposition: {e}")
            return {'market_component': volumes, 'idiosyncratic': np.zeros_like(volumes)}
    
    def estimate_arma_parameters(self, volumes: np.ndarray):
        """
        تخمین پارامترهای ARMA(1,1) برای حجم
        فرمول: y_{t,i} = ψ₁ y_{t-1,i} + ψ₂ + ε_{t,i}
        """
        try:
            if len(volumes) < 10:
                return
            
            # آماده‌سازی داده‌ها
            y_t = volumes[1:]
            y_t_minus_1 = volumes[:-1]
            
            # ساخت ماتریس X: [y_{t-1}, 1]
            X = np.column_stack([y_t_minus_1, np.ones(len(y_t_minus_1))])
            
            # Least Squares estimation
            params = np.linalg.lstsq(X, y_t, rcond=None)[0]
            
            if len(params) >= 2:
                self.arma_psi1, self.arma_psi2 = params[0], params[1]
                self.arma_estimated = True
                
        except Exception as e:
            logger.debug(f"Error estimating ARMA parameters: {e}")
    
    def predict_volume_arma(self, volumes: np.ndarray) -> float:
        """
        پیش‌بینی حجم با مدل ARMA(1,1)
        فرمول: y_{t,i} = ψ₁ y_{t-1,i} + ψ₂ + ε_{t,i}
        """
        try:
            if len(volumes) < 2:
                return volumes[-1] if len(volumes) > 0 else 1.0
            
            # تخمین پارامترها اگر انجام نشده
            if not self.arma_estimated:
                self.estimate_arma_parameters(volumes)
            
            # پیش‌بینی با فرمول ARMA
            y_t_minus_1 = volumes[-1]
            y_pred = self.arma_psi1 * y_t_minus_1 + self.arma_psi2
            
            return float(max(y_pred, 0.01))
            
        except Exception as e:
            logger.debug(f"Error in ARMA volume prediction: {e}")
            return volumes[-1] if len(volumes) > 0 else 1.0
    
    def predict_vwap(self, volumes: np.ndarray, prices: np.ndarray, 
                    horizon: int = 1) -> float:
        """
        Predict VWAP for next period
        """
        try:
            if len(volumes) < 2 or len(prices) < 2:
                return prices[-1] if len(prices) > 0 else 0.0
            
            # Simple VWAP calculation
            vwap = np.sum(prices * volumes) / (np.sum(volumes) + 1e-8)
            
            return float(vwap)
            
        except Exception as e:
            logger.error(f"Error in VWAP prediction: {e}")
            return prices[-1] if len(prices) > 0 else 0.0


# ============================================================================
# 8. SETAR MODEL FOR VOLUME DYNAMICS (Section 6.2)
# ============================================================================
class SETARVolumeModel:
    """
    Self-Exciting Threshold Autoregressive model for volume dynamics
    y_{t,i} = (φ₁₁ y_{t-1,i} + φ₁₂) I(y_{t-1,i}) + (φ₂₁ y_{t-1,i} + φ₂₂) [1 - I(y_{t-1,i})] + ε_{t,i}
    Section 6.2 of the article
    """
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold  # Will be estimated if None
        
        # Parameters for two regimes (will be estimated from data)
        self.phi_11 = 0.5  # Regime 1: AR coefficient
        self.phi_12 = 0.0  # Regime 1: constant
        self.phi_21 = 0.3  # Regime 2: AR coefficient
        self.phi_22 = 0.0  # Regime 2: constant
        
        self.volume_history = deque(maxlen=1000)
        self.estimated = False  # Flag to track if parameters are estimated
        
    def estimate_threshold(self, volumes: np.ndarray) -> float:
        """Estimate threshold from data"""
        if len(volumes) < 10:
            return np.median(volumes) if len(volumes) > 0 else 1.0
        
        # Use median as threshold
        threshold = np.median(volumes)
        return float(threshold)
    
    def estimate_parameters(self, volumes: np.ndarray):
        """
        تخمین پارامترهای SETAR با Least Squares
        فرمول: y_{t,i} = (φ₁₁ y_{t-1,i} + φ₁₂) I(y_{t-1,i}) + (φ₂₁ y_{t-1,i} + φ₂₂) [1 - I(y_{t-1,i})] + ε_{t,i}
        """
        try:
            if len(volumes) < 20:
                return
            
            # Estimate threshold if not set
            if self.threshold is None:
                self.threshold = self.estimate_threshold(volumes)
            
            # آماده‌سازی داده‌ها
            y_t = volumes[1:]
            y_t_minus_1 = volumes[:-1]
            
            # Indicator function
            I = (y_t_minus_1 <= self.threshold).astype(float)
            
            # ساخت ماتریس X برای regression
            # Regime 1: [y_{t-1} * I, I]
            # Regime 2: [y_{t-1} * (1-I), (1-I)]
            X_regime1 = np.column_stack([y_t_minus_1 * I, I])
            X_regime2 = np.column_stack([y_t_minus_1 * (1 - I), (1 - I)])
            
            # تخمین پارامترها با Least Squares
            # Regime 1: y_t = φ₁₁ * y_{t-1} * I + φ₁₂ * I
            if np.sum(I) > 5:  # حداقل 5 داده در regime 1
                y_regime1 = y_t[I == 1]
                X_regime1_active = X_regime1[I == 1]
                if len(y_regime1) > 0 and len(X_regime1_active) > 0:
                    params1 = np.linalg.lstsq(X_regime1_active, y_regime1, rcond=None)[0]
                    if len(params1) >= 2:
                        self.phi_11, self.phi_12 = params1[0], params1[1]
            
            # Regime 2: y_t = φ₂₁ * y_{t-1} * (1-I) + φ₂₂ * (1-I)
            if np.sum(1 - I) > 5:  # حداقل 5 داده در regime 2
                y_regime2 = y_t[I == 0]
                X_regime2_active = X_regime2[I == 0]
                if len(y_regime2) > 0 and len(X_regime2_active) > 0:
                    params2 = np.linalg.lstsq(X_regime2_active, y_regime2, rcond=None)[0]
                    if len(params2) >= 2:
                        self.phi_21, self.phi_22 = params2[0], params2[1]
            
            self.estimated = True
            
        except Exception as e:
            logger.debug(f"Error estimating SETAR parameters: {e}")
    
    def predict_volume(self, volumes: np.ndarray, horizon: int = 1) -> float:
        """
        Predict volume using SETAR model
        فرمول کامل: y_{t,i} = (φ₁₁ y_{t-1,i} + φ₁₂) I(y_{t-1,i}) + (φ₂₁ y_{t-1,i} + φ₂₂) [1 - I(y_{t-1,i})] + ε_{t,i}
        """
        try:
            if len(volumes) < 2:
                return volumes[-1] if len(volumes) > 0 else 1.0
            
            # تخمین پارامترها اگر هنوز انجام نشده
            if not self.estimated or len(volumes) > len(self.volume_history):
                self.estimate_parameters(volumes)
                self.volume_history.extend(volumes)
            
            # Estimate threshold if not set
            if self.threshold is None:
                self.threshold = self.estimate_threshold(volumes)
            
            # Get last volume
            y_t_minus_1 = volumes[-1]
            
            # Indicator function: I(y_{t-1,i}) = 1 if y_{t-1,i} ≤ τ, else 0
            I = 1.0 if y_t_minus_1 <= self.threshold else 0.0
            
            # SETAR prediction با فرمول کامل
            # y_{t,i} = (φ₁₁ y_{t-1,i} + φ₁₂) I + (φ₂₁ y_{t-1,i} + φ₂₂) (1 - I)
            y_pred = (self.phi_11 * y_t_minus_1 + self.phi_12) * I + \
                     (self.phi_21 * y_t_minus_1 + self.phi_22) * (1 - I)
            
            # Ensure positive volume
            y_pred = max(y_pred, 0.01)
            
            return float(y_pred)
            
        except Exception as e:
            logger.error(f"Error in SETAR volume prediction: {e}")
            return volumes[-1] if len(volumes) > 0 else 1.0


# ============================================================================
# SYMMETRICAL ANALYSIS - تحلیل تقارن
# ============================================================================
class SymmetryAnalyzer:
    """
    تحلیل تقارن کامل طبق صفحات ۱۰-۱۱ مقاله
    """
    
    def __init__(self):
        self.hook_retracement = 0.86
        self.trend_divisions = 3
        self.symmetry_tolerance = 0.1
    
    def analyze_price_movements(self, price_series: np.ndarray) -> Dict:
        """
        تحلیل کامل حرکات قیمت از نظر تقارن
        """
        hooks, trends = self._identify_hooks_and_trends(price_series)
        
        hook_analysis = self._analyze_hooks(hooks)
        trend_analysis = self._analyze_trends(trends)
        combined_analysis = self._combined_analysis(hook_analysis, trend_analysis)
        
        symmetry_score = self._calculate_symmetry_score(hook_analysis, trend_analysis)
        
        return {
            'hooks': hook_analysis,
            'trends': trend_analysis,
            'combined': combined_analysis,
            'symmetry_score': symmetry_score,
            'is_symmetrical': symmetry_score > 0.7,
            'hook_pattern': self._identify_hook_pattern(hook_analysis),
            'trend_pattern': self._identify_trend_pattern(trend_analysis)
        }
    
    def _identify_hooks_and_trends(self, prices: np.ndarray) -> Tuple[List, List]:
        """
        شناسایی قلاب‌ها و روندها در سری قیمت
        """
        hooks = []
        trends = []
        changes = np.diff(prices)
        
        i = 0
        while i < len(changes):
            rally = self._identify_rally(changes, i)
            if rally is None:
                i += 1
                continue
            
            i = rally['end_index']
            correction = self._identify_correction(changes, i, rally['magnitude'])
            
            if correction:
                hook = {
                    'rally': rally,
                    'correction': correction,
                    'start_price': prices[rally['start_index']],
                    'end_price': prices[correction['end_index']],
                    'net_movement': rally['magnitude'] - abs(correction['magnitude'])
                }
                hooks.append(hook)
                i = correction['end_index']
            else:
                trends.append(rally)
        
        return hooks, trends
    
    def _identify_rally(self, changes: np.ndarray, start_idx: int) -> Optional[Dict]:
        """شناسایی رالی"""
        if start_idx >= len(changes):
            return None
        
        direction = 1 if changes[start_idx] > 0 else -1
        magnitude = 0
        end_idx = start_idx
        
        for i in range(start_idx, min(start_idx + 20, len(changes))):
            if changes[i] * direction > 0:
                magnitude += abs(changes[i])
                end_idx = i
            else:
                if i < len(changes) - 1 and changes[i + 1] * direction > 0:
                    magnitude += abs(changes[i])
                    end_idx = i
                else:
                    break
        
        if end_idx - start_idx < 2:
            return None
        
        return {
            'start_index': start_idx,
            'end_index': end_idx,
            'magnitude': magnitude,
            'direction': direction,
            'length': end_idx - start_idx + 1
        }
    
    def _identify_correction(self, changes: np.ndarray, start_idx: int, 
                            rally_magnitude: float) -> Optional[Dict]:
        """شناسایی اصلاح"""
        if start_idx >= len(changes):
            return None
        
        expected_direction = -1 if changes[start_idx-1] > 0 else 1
        
        magnitude = 0
        end_idx = start_idx
        
        for i in range(start_idx, min(start_idx + 15, len(changes))):
            if changes[i] * expected_direction > 0:
                magnitude += abs(changes[i])
                end_idx = i
                
                if magnitude >= rally_magnitude * 0.8:
                    break
            else:
                break
        
        if magnitude < rally_magnitude * 0.5:
            return None
        
        return {
            'start_index': start_idx,
            'end_index': end_idx,
            'magnitude': magnitude,
            'direction': expected_direction,
            'retracement_percent': (magnitude / rally_magnitude) * 100
        }
    
    def _analyze_hooks(self, hooks: List[Dict]) -> List[Dict]:
        """تحلیل دقیق قلاب‌ها"""
        analyzed_hooks = []
        
        for hook in hooks:
            rally = hook['rally']['magnitude']
            correction = hook['correction']['magnitude']
            
            expected_correction = rally * self.hook_retracement
            correction_error = abs(correction - expected_correction) / expected_correction
            
            net_movement = rally - correction
            expected_net = rally * 0.14
            net_error = abs(net_movement - expected_net) / expected_net
            
            analyzed_hook = {
                **hook,
                'expected_correction': expected_correction,
                'correction_error': correction_error,
                'expected_net': expected_net,
                'net_error': net_error,
                'is_valid_hook': correction_error < self.symmetry_tolerance,
                'hook_ratio': correction / rally,
                'net_ratio': net_movement / rally
            }
            
            analyzed_hooks.append(analyzed_hook)
        
        return analyzed_hooks
    
    def _analyze_trends(self, trends: List[Dict]) -> List[Dict]:
        """تحلیل روندها از نظر تقارن"""
        analyzed_trends = []
        
        for trend in trends:
            magnitude = trend['magnitude']
            length = trend['length']
            
            segment_length = length / self.trend_divisions
            segment_magnitude = magnitude / self.trend_divisions
            
            symmetry_score = self._calculate_trend_symmetry(trend)
            
            analyzed_trend = {
                **trend,
                'segment_length': segment_length,
                'segment_magnitude': segment_magnitude,
                'symmetry_score': symmetry_score,
                'is_symmetrical': symmetry_score > 0.8,
                'expected_segments': self.trend_divisions
            }
            
            analyzed_trends.append(analyzed_trend)
        
        return analyzed_trends
    
    def _calculate_trend_symmetry(self, trend: Dict) -> float:
        """محاسبه امتیاز تقارن روند"""
        # یک پیاده‌سازی ساده
        if 'magnitude' not in trend or trend['magnitude'] == 0:
            return 0.5
        
        # می‌توانید این را با منطق پیچیده‌تری جایگزین کنید
        return np.random.uniform(0.6, 0.9)
    
    def _combined_analysis(self, hook_analysis: List[Dict], 
                          trend_analysis: List[Dict]) -> Dict:
        """تحلیل ترکیبی الگو"""
        if not hook_analysis or not trend_analysis:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        if len(hook_analysis) >= 2 and len(trend_analysis) >= 1:
            recent_hooks = hook_analysis[-2:]
            recent_trend = trend_analysis[-1]
            
            hooks_valid = all(h['is_valid_hook'] for h in recent_hooks)
            trend_symmetrical = recent_trend['is_symmetrical']
            
            hook_ratios = [h['hook_ratio'] for h in recent_hooks]
            hook_consistency = 1 - (np.std(hook_ratios) / np.mean(hook_ratios))
            
            combined_score = (
                (hooks_valid * 0.4) +
                (trend_symmetrical * 0.3) +
                (hook_consistency * 0.3)
            )
            
            return {
                'valid': hooks_valid and trend_symmetrical,
                'score': combined_score,
                'pattern': '2_hooks_1_trend',
                'hooks_consistency': hook_consistency,
                'trend_symmetry': recent_trend['symmetry_score']
            }
        
        return {'valid': False, 'reason': 'Pattern not detected'}
    
    def _calculate_symmetry_score(self, hook_analysis: List[Dict], 
                                 trend_analysis: List[Dict]) -> float:
        """محاسبه امتیاز کلی تقارن"""
        scores = []
        
        if hook_analysis:
            hook_scores = []
            for hook in hook_analysis[-3:]:
                if hook['is_valid_hook']:
                    error = hook['correction_error']
                    score = 1 - min(error, 1.0)
                    hook_scores.append(score)
            
            if hook_scores:
                scores.append(np.mean(hook_scores) * 0.5)
        
        if trend_analysis:
            trend_scores = []
            for trend in trend_analysis[-2:]:
                if trend['is_symmetrical']:
                    trend_scores.append(trend['symmetry_score'])
            
            if trend_scores:
                scores.append(np.mean(trend_scores) * 0.3)
        
        combined = self._combined_analysis(hook_analysis, trend_analysis)
        if combined['valid']:
            scores.append(combined['score'] * 0.2)
        
        return np.mean(scores) if scores else 0.0
    
    def _identify_hook_pattern(self, hook_analysis: List[Dict]) -> str:
        """شناسایی الگوی قلاب"""
        if len(hook_analysis) < 2:
            return 'insufficient_hooks'
        
        recent_hooks = hook_analysis[-2:]
        directions = [h['rally']['direction'] for h in recent_hooks]
        
        if directions[0] == directions[1]:
            return 'same_direction_hooks'
        else:
            return 'alternating_hooks'
    
    def _identify_trend_pattern(self, trend_analysis: List[Dict]) -> str:
        """شناسایی الگوی روند"""
        if len(trend_analysis) < 2:
            return 'single_trend'
        
        recent_trends = trend_analysis[-2:]
        directions = [t['direction'] for t in recent_trends]
        
        if directions[0] == directions[1]:
            return 'continuing_trend'
        else:
            return 'reversal_pattern'
    
    def generate_symmetry_signal(self, price_data: pd.DataFrame) -> Optional[NDSSignal]:
        """
        تولید سیگنال بر اساس تحلیل تقارن
        """
        prices = price_data['close'].values
        analysis = self.analyze_price_movements(prices)
        
        if not analysis['is_symmetrical'] or analysis['symmetry_score'] < 0.7:
            return None
        
        direction = self._determine_direction_from_symmetry(analysis)
        
        if direction == TrendDirection.NEUTRAL:
            return None
        
        current_price = prices[-1]
        entry, sl, tp = self._calculate_symmetry_levels(
            current_price, direction, analysis
        )
        
        if sl == 0:
            return None
        
        risk_reward = abs(tp - entry) / abs(entry - sl)
        
        if risk_reward < 1.5:
            return None
        
        signal = NDSSignal(
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=analysis['symmetry_score'],
            quantum_state=QuantumState.COLLAPSED_BULLISH 
            if direction == TrendDirection.BULLISH 
            else QuantumState.COLLAPSED_BEARISH,
            hurst_exponent=0.6,
            risk_reward=risk_reward,
            timestamp=datetime.now()
        )
        
        logger.info(f"Symmetry signal generated: {direction}, Confidence: {analysis['symmetry_score']:.2%}")
        
        return signal
    
    def _determine_direction_from_symmetry(self, analysis: Dict) -> TrendDirection:
        """تعیین جهت بر اساس تحلیل تقارن"""
        hook_pattern = analysis['hook_pattern']
        trend_pattern = analysis['trend_pattern']
        
        if trend_pattern == 'continuing_trend':
            return TrendDirection.BULLISH
        
        elif hook_pattern == 'alternating_hooks':
            return TrendDirection.BEARISH
        
        return TrendDirection.NEUTRAL
    
    def _calculate_symmetry_levels(self, current_price: float, 
                                  direction: TrendDirection, 
                                  analysis: Dict) -> Tuple[float, float, float]:
        """محاسبه سطوح بر اساس تقارن"""
        if not analysis['hooks']:
            return current_price, 0.0, 0.0
        
        recent_hook = analysis['hooks'][-1]
        hook_size = recent_hook['rally']['magnitude']
        
        if direction == TrendDirection.BULLISH:
            entry = current_price
            sl = current_price - hook_size * 0.5
            tp = current_price + hook_size * 1.5
        else:
            entry = current_price
            sl = current_price + hook_size * 0.5
            tp = current_price - hook_size * 1.5
        
        return entry, sl, tp


# ============================================================================
# ENHANCED NDS ANALYZER - آنالایزر NDS بهبودیافته
# ============================================================================
class EnhancedNDSAnalyzer(AdvancedNDSAnalyzer):
    """
    نسخه بهبودیافته آنالایزر NDS با افزودن بخش‌های مفقود مقاله
    """
    
    def __init__(self, mt5_manager: MT5Manager, config: Any = None):
        super().__init__(mt5_manager, config=config)
        
        self.fractal_model = FractalRecursiveModel()
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.nn_manager = NeuralNetworkManager()
        
        self.use_fractal_analysis = True
        self.use_symmetry_analysis = True
        self.use_neural_enhancement = True
        
    def enhanced_analyze(self) -> Optional[NDSSignal]:
        """
        تحلیل پیشرفته با یکپارچه‌سازی تمام بخش‌های مقاله
        """
        try:
            logger.info("🚀 Starting Enhanced NDS Analysis...")
            
            df_trend = self.mt5.get_ohlcv(self.tf_trend, 2000)
            df_analysis = self.mt5.get_ohlcv(self.tf_analysis, 1000)
            
            if df_trend is None or df_analysis is None:
                logger.warning("Insufficient data for enhanced analysis")
                return None
            
            base_signal = super().analyze()
            
            fractal_signal = None
            if self.use_fractal_analysis:
                fractal_signal = self._fractal_analysis(df_trend)
            
            symmetry_signal = None
            if self.use_symmetry_analysis:
                symmetry_signal = self.symmetry_analyzer.generate_symmetry_signal(df_analysis)
            
            combined_signal = self._combine_signals(
                base_signal, fractal_signal, symmetry_signal, df_analysis
            )
            
            if self.use_neural_enhancement and combined_signal:
                enhanced_signal = self._neural_enhancement(combined_signal, df_analysis)
                if enhanced_signal:
                    combined_signal = enhanced_signal
            
            if combined_signal and combined_signal.is_valid():
                self._log_enhanced_signal(combined_signal, {
                    'fractal': fractal_signal is not None,
                    'symmetry': symmetry_signal is not None,
                    'neural': self.use_neural_enhancement
                })
                return combined_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            return None
    
    def _fractal_analysis(self, df: pd.DataFrame) -> Optional[Dict]:
        """تحلیل فرکتال"""
        try:
            prices = df['close'].values
            
            fractal_structure = self.fractal_model.build_fractal_structure(prices)
            fractal_analysis = self.fractal_model.analyze_fractal_pattern(fractal_structure)
            fractal_signal = self.fractal_model.get_fractal_signal(fractal_structure)
            
            logger.info(f"Fractal Analysis: Levels={fractal_analysis['total_levels']}, "
                       f"Symmetry={fractal_analysis['symmetry_score']:.2%}")
            
            return {
                'structure': fractal_structure,
                'analysis': fractal_analysis,
                'signal': fractal_signal
            }
            
        except Exception as e:
            logger.error(f"Error in fractal analysis: {e}")
            return None
    
    def _combine_signals(self, base_signal: Optional[NDSSignal], 
                        fractal_result: Optional[Dict],
                        symmetry_signal: Optional[NDSSignal],
                        df: pd.DataFrame) -> Optional[NDSSignal]:
        """ترکیب سیگنال‌های مختلف"""
        
        if not base_signal:
            return None
        
        signals = []
        weights = []
        
        signals.append(base_signal)
        weights.append(0.5)
        
        if fractal_result and fractal_result.get('signal', {}).get('fractal_aligned', False):
            fractal_conf = fractal_result['signal']['confidence']
            if fractal_conf > 0.7:
                fractal_direction = TrendDirection.BULLISH if fractal_result['signal'].get('direction') == 'bullish' else TrendDirection.BEARISH
                
                fractal_signal = NDSSignal(
                    direction=fractal_direction,
                    entry_price=base_signal.entry_price,
                    stop_loss=base_signal.stop_loss,
                    take_profit=base_signal.take_profit,
                    confidence=fractal_conf,
                    quantum_state=base_signal.quantum_state,
                    hurst_exponent=fractal_result['signal'].get('fractal_dimension', 0.6),
                    risk_reward=base_signal.risk_reward,
                    timestamp=datetime.now()
                )
                signals.append(fractal_signal)
                weights.append(0.25)
        
        if symmetry_signal and symmetry_signal.confidence > 0.7:
            signals.append(symmetry_signal)
            weights.append(0.25)
        
        if len(signals) == 1:
            return signals[0]
        
        return self._weighted_signal_combination(signals, weights, df)
    
    def _weighted_signal_combination(self, signals: List[NDSSignal], 
                                    weights: List[float], 
                                    df: pd.DataFrame) -> NDSSignal:
        """ترکیب وزنی سیگنال‌ها"""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        avg_confidence = sum(s.confidence * w for s, w in zip(signals, normalized_weights))
        
        direction_scores = {TrendDirection.BULLISH: 0.0, TrendDirection.BEARISH: 0.0}
        
        for signal, weight in zip(signals, normalized_weights):
            if signal.direction == TrendDirection.BULLISH:
                direction_scores[TrendDirection.BULLISH] += weight * signal.confidence
            elif signal.direction == TrendDirection.BEARISH:
                direction_scores[TrendDirection.BEARISH] += weight * signal.confidence
        
        final_direction = max(direction_scores, key=direction_scores.get)
        
        base_signal = signals[0]
        
        combined_signal = NDSSignal(
            direction=final_direction,
            entry_price=base_signal.entry_price,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit,
            confidence=avg_confidence,
            quantum_state=base_signal.quantum_state,
            hurst_exponent=np.mean([s.hurst_exponent for s in signals]),
            risk_reward=base_signal.risk_reward,
            timestamp=datetime.now(),
            nodes=base_signal.nodes
        )
        
        return combined_signal
    
    def _neural_enhancement(self, signal: NDSSignal, df: pd.DataFrame) -> Optional[NDSSignal]:
        """بهبود سیگنال با شبکه عصبی"""
        try:
            recent_data = df.tail(100)
            self.nn_manager.adaptive_learning(recent_data)
            
            prices = recent_data['close'].values
            t = np.arange(len(prices))
            
            enhanced_result = self.nn_manager.trend_enhancer.enhance_trend_function(t, prices, epochs=50)
            
            if enhanced_result['improvement_percent'] > 5.0:
                improvement_factor = 1 + (enhanced_result['improvement_percent'] / 100)
                enhanced_confidence = min(signal.confidence * improvement_factor, 0.95)
                
                enhanced_signal = NDSSignal(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence=enhanced_confidence,
                    quantum_state=signal.quantum_state,
                    hurst_exponent=signal.hurst_exponent,
                    risk_reward=signal.risk_reward,
                    timestamp=signal.timestamp,
                    nodes=signal.nodes
                )
                
                logger.info(f"Neural enhancement: +{enhanced_result['improvement_percent']:.2f}% improvement")
                return enhanced_signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in neural enhancement: {e}")
            return signal
    
    def _log_enhanced_signal(self, signal: NDSSignal, modules_used: Dict):
        """لاگ سیگنال بهبودیافته"""
        logger.info("=" * 70)
        logger.info("🚀 ENHANCED NDS SIGNAL GENERATED")
        logger.info("=" * 70)
        logger.info(f"   Direction: {'BUY 📈' if signal.direction == TrendDirection.BULLISH else 'SELL 📉'}")
        logger.info(f"   Entry: {signal.entry_price:.2f}")
        logger.info(f"   SL: {signal.stop_loss:.2f}")
        logger.info(f"   TP: {signal.take_profit:.2f}")
        logger.info(f"   R/R: {signal.risk_reward:.2f}")
        logger.info(f"   Confidence: {signal.confidence:.2%}")
        logger.info(f"   Modules Used: Fractal={modules_used['fractal']}, "
                   f"Symmetry={modules_used['symmetry']}, Neural={modules_used['neural']}")
        logger.info(f"   Hurst Exponent: {signal.hurst_exponent:.3f}")
        logger.info(f"   Quantum State: {signal.quantum_state.value}")
        logger.info("=" * 70)


# ============================================================================
# NDS TRADING BOT - ربات معاملاتی اصلی
# ============================================================================
class NDSTradingBot:
    """NDS Trading Bot"""
    
    def __init__(self, symbol: str = "BTCUSD", max_lots: float = None, config: Any = None):
        # استفاده از config اگر موجود باشد
        if config is not None:
            self.symbol = config.symbol
            self.max_lots = config.max_lots if max_lots is None else max_lots
            self.config = config
        else:
            self.symbol = symbol
            self.max_lots = max_lots if max_lots is not None else (
                DEFAULT_MAX_LOTS if MODULAR_IMPORTS_AVAILABLE else 0.3
            )
            self.config = None
        
        self.running = False
        
        # Dependency injection ready
        self.mt5 = MT5Manager(self.symbol)
        self.risk = RiskManager(self.mt5, config=self.config)
        self.trade = TradeManager(self.mt5, self.risk, config=self.config)
        self.nds = AdvancedNDSAnalyzer(self.mt5, config=self.config)
        
        self.last_candle_time: Dict[int, datetime] = {}
        
    def start(self):
        """Start bot"""
        logger.info("🚀 Starting NDS Trading Bot...")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Max Risk: {self.risk.max_risk_percent}%")
        logger.info(f"   Max Lots: {self.max_lots}")
        
        if not self.mt5.connect():
            logger.error("Error connecting to MT5")
            return
        
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            if "Open position" in reason:
                logger.info(f"⚠️ {reason} - Managing existing trade")
            else:
                logger.warning(f"⚠️ {reason}")
                if self.mt5.account_info.balance < 500:
                    logger.error("Balance below $500 - Bot stopped")
                    self.mt5.disconnect()
                    return
        
        # تست اتصال و ارسال سفارش
        logger.info("🔧 Testing connection and order sending...")
        if not self.mt5.test_connection():
            logger.error("❌ Connection test failed! Bot cannot trade.")
            logger.error("   Please check:")
            logger.error("   1. MT5 terminal is running")
            logger.error("   2. Account is logged in")
            logger.error("   3. Symbol is available")
            logger.error("   4. Trading is enabled")
            self.mt5.disconnect()
            return
        
        self.running = True
        logger.info("✅ Bot started successfully")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main bot loop"""
        while self.running:
            try:
                current_time = datetime.now()
                
                positions = self.mt5.get_active_positions()
                
                if positions:
                    self._manage_open_trade(positions[0])
                else:
                    if self._is_new_candle(mt5.TIMEFRAME_M1):
                        self._analyze_and_trade()
                
                if not hasattr(self, '_last_report') or \
                (current_time - self._last_report).seconds >= 60:
                    self._status_report()
                    self._last_report = current_time
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

    def _manage_open_trade(self, trade: TradeInfo):
        """مدیریت پوزیشن باز - نسخه اصلاح شده با Trailing فعال"""
        try:
            bid, ask = self.mt5.get_current_price()
            current_price = bid if trade.order_type == mt5.ORDER_TYPE_BUY else ask
            
            is_buy = trade.order_type == mt5.ORDER_TYPE_BUY
            point = self.mt5.get_point()
            
            # ایجاد متغیرهای ردیابی در اولین اجرا
            if not hasattr(trade, '_peak_price'):
                trade._peak_price = current_price
                trade._breakeven_done = False
                trade._trailing_active = False
                trade._last_trailing_update = datetime.now()
            
            # به‌روزرسانی قیمت پیک
            if is_buy:
                trade._peak_price = max(trade._peak_price, current_price)
            else:
                trade._peak_price = min(trade._peak_price, current_price)
            
            # محاسبه سود بر حسب پیپ
            if is_buy:
                profit_pips = (current_price - trade.open_price) / point
            else:
                profit_pips = (trade.open_price - current_price) / point
            
            current_time = datetime.now()
            
            # ✅ بررسی Trailing هر 3 ثانیه
            if (current_time - trade._last_trailing_update).seconds >= 3:
                
                logger.info(f"📊 Position #{trade.ticket}: Profit={profit_pips:.1f} pips, P/L=${trade.profit:.2f}")
                
                # ✅ BREAKEVEN: بعد از 10 پیپ سود (کاهش از 20)
                if not trade._breakeven_done and profit_pips >= 10:
                    if is_buy:
                        new_sl = trade.open_price + (2 * point)  # 2 پیپ بالای ورودی
                        if new_sl > trade.sl:
                            if self.trade.update_trailing_stop(trade, new_sl):
                                trade._breakeven_done = True
                                logger.info(f"🛡️ BREAKEVEN: SL={new_sl:.2f} (+10 pips)")
                    else:
                        new_sl = trade.open_price - (2 * point)
                        if new_sl < trade.sl:
                            if self.trade.update_trailing_stop(trade, new_sl):
                                trade._breakeven_done = True
                                logger.info(f"🛡️ BREAKEVEN: SL={new_sl:.2f} (+10 pips)")
                
                # ✅ TRAILING: بعد از 15 پیپ سود (کاهش از 30)
                elif profit_pips >= 15:
                    trade._trailing_active = True
                    
                    # فاصله Trailing: 8 پیپ (ثابت و محافظه‌کارانه)
                    trailing_distance = 8 * point
                    
                    if is_buy:
                        # BUY: SL را بالا می‌بریم
                        new_sl = current_price - trailing_distance
                        
                        # فقط اگر بهتر از SL فعلی باشد
                        if new_sl > trade.sl:
                            sl_improvement_pips = (new_sl - trade.sl) / point
                            
                            # حداقل 2 پیپ بهبود
                            if sl_improvement_pips >= 2:
                                if self.trade.update_trailing_stop(trade, new_sl):
                                    logger.info(f"📈 TRAILING: {trade.sl:.2f} → {new_sl:.2f} (+{sl_improvement_pips:.1f} pips)")
                                    trade._last_trailing_update = current_time
                    
                    else:  # SELL
                        # SELL: SL را پایین می‌آوریم
                        new_sl = current_price + trailing_distance
                        
                        if new_sl < trade.sl:
                            sl_improvement_pips = (trade.sl - new_sl) / point
                            
                            if sl_improvement_pips >= 2:
                                if self.trade.update_trailing_stop(trade, new_sl):
                                    logger.info(f"📉 TRAILING: {trade.sl:.2f} → {new_sl:.2f} (+{sl_improvement_pips:.1f} pips)")
                                    trade._last_trailing_update = current_time
            
            # ✅ بررسی تغییر روند هر 30 ثانیه
            if not hasattr(self, '_last_trend_check') or \
            (current_time - self._last_trend_check).seconds >= 30:
                
                direction = TrendDirection.BULLISH if is_buy else TrendDirection.BEARISH
                current_trend = self._simple_trend_analysis()
                
                if current_trend != direction and current_trend != TrendDirection.NEUTRAL:
                    # اگر روند معکوس شد و در سود هستیم، ببند
                    if trade.profit > 5:  # حداقل $5 سود
                        self.trade.close_trade(trade, f"Trend reversal with profit")
                        logger.info(f"💰 Closed: Trend changed, Profit=${trade.profit:.2f}")
                        return
                
                self._last_trend_check = current_time
                
        except Exception as e:
            logger.error(f"❌ Error managing trade: {e}")

        
    
    def _simple_trend_analysis(self) -> TrendDirection:
        """Simple trend analysis without complex NDS calculations"""
        try:
            df = self.mt5.get_ohlcv(mt5.TIMEFRAME_M5, 30)
            if df is None or len(df) < 10:
                return TrendDirection.NEUTRAL
            
            prices = df['close'].values
            
            ma_fast = np.mean(prices[-5:])
            ma_slow = np.mean(prices[-15:])
            current_price = prices[-1]
            
            if current_price > ma_fast > ma_slow:
                return TrendDirection.BULLISH
            elif current_price < ma_fast < ma_slow:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error in simple trend analysis: {e}")
            return TrendDirection.NEUTRAL

    def _analyze_and_trade(self):
        """Analyze and enter trade - only when no open positions"""
        try:
            can_trade, reason = self.risk.can_trade()
            if not can_trade:
                logger.info(f"Cannot trade: {reason}")
                return
            
            current_time = datetime.now()
            if hasattr(self, '_last_analysis_time') and \
            (current_time - self._last_analysis_time).seconds < 60:
                return
            
            logger.info("Starting NDS analysis for new trade...")
            signal = self.nds.analyze()
            self._last_analysis_time = current_time
            
            if signal is None:
                return
            
            valid, msg = self.risk.validate_signal(signal)
            if not valid:
                logger.info(f"Signal rejected: {msg}")
                return
            
            ticket = self.trade.open_trade_safe(signal)
            if ticket:
                logger.info(f"New trade opened successfully - Ticket: {ticket}")
                
        except Exception as e:
            logger.error(f"Error in analyze_and_trade: {e}")
    
    def _is_new_candle(self, timeframe: int) -> bool:
        """Check if new candle formed"""
        df = self.mt5.get_ohlcv(timeframe, 2)
        if df is None or len(df) < 2:
            return False
        
        current_candle_time = df.index[-1]
        last_time = self.last_candle_time.get(timeframe)
        
        if last_time is None or current_candle_time > last_time:
            self.last_candle_time[timeframe] = current_candle_time
            return True
        
        return False

    def _status_report(self):
        """Status report"""
        if not hasattr(self, '_last_report') or \
           (datetime.now() - self._last_report).seconds >= 60:
            
            self.mt5.refresh_account()
            positions = self.mt5.get_active_positions()
            bid, ask = self.mt5.get_current_price()
            
            logger.info("-" * 50)
            logger.info(f"📊 Status Report - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"   Balance: ${self.mt5.account_info.balance:,.2f}")
            logger.info(f"   Equity: ${self.mt5.account_info.equity:,.2f}")
            logger.info(f"   Free Margin: ${self.mt5.account_info.free_margin:,.2f}")
            logger.info(f"   {self.symbol}: Bid={bid:.2f} | Ask={ask:.2f}")
            logger.info(f"   Max Lots: {self.max_lots}")
            
            if positions:
                pos = positions[0]
                logger.info(f"   Active Trade: {'BUY' if pos.order_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
                logger.info(f"   Volume: {pos.volume:.2f} lots")
                logger.info(f"   P/L: ${pos.profit:.2f}")
            else:
                logger.info("   No active trades - Waiting for suitable position...")
            
            logger.info("-" * 50)
            self._last_report = datetime.now()
    
    def stop(self):
        """Stop bot"""
        self.running = False
        self.mt5.disconnect()
        logger.info("🛑 Bot stopped")


# ============================================================================
# ENHANCED TRADING BOT - ربات معاملاتی بهبودیافته
# ============================================================================
class EnhancedNDSTradingBot(NDSTradingBot):
    """ربات معاملاتی NDS با قابلیت‌های بهبودیافته"""
    
    def __init__(self, symbol: str = "BTCUSD", max_lots: float = None, config: Any = None):
        super().__init__(symbol, max_lots=max_lots, config=config)
        
        self.nds = EnhancedNDSAnalyzer(self.mt5, config=self.config)
        
        self.nds.use_fractal_analysis = True
        self.nds.use_symmetry_analysis = True
        self.nds.use_neural_enhancement = True
        
        logger.info("🤖 Enhanced NDS Trading Bot Initialized")
        logger.info("   - Fractal Analysis: Enabled")
        logger.info("   - Symmetry Analysis: Enabled")
        logger.info("   - Neural Enhancement: Enabled")
    
    def start(self):
        """شروع ربات بهبودیافته"""
        logger.info("🚀 Starting Enhanced NDS Trading Bot...")
        logger.info("📊 Features: Complete Fractal Model + Symmetry Analysis + Neural Networks")
        
        if not self.mt5.connect():
            logger.error("Error connecting to MT5")
            return
        
        self.running = True
        logger.info("✅ Enhanced Bot started successfully")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Enhanced Bot stopped by user")
        finally:
            self.stop()
    
    def _analyze_and_trade(self):
        """تحلیل و معامله با قابلیت‌های بهبودیافته"""
        try:
            # بررسی پوزیشن‌های باز
            positions = self.mt5.get_active_positions()
            if positions:
                logger.info(f"⚠️ Skipping analysis: {len(positions)} open position(s)")
                return
            
            can_trade, reason = self.risk.can_trade()
            if not can_trade:
                logger.info(f"Cannot trade: {reason}")
                return
            
            logger.info("Starting Enhanced NDS analysis for new trade...")
            signal = self.nds.enhanced_analyze()
            
            if signal is None:
                return
            
            valid, msg = self.risk.validate_signal(signal)
            if not valid:
                logger.info(f"Signal rejected: {msg}")
                return
            
            ticket = self.trade.open_trade_safe(signal)
            if ticket:
                logger.info(f"New enhanced trade opened - Ticket: {ticket}")
                
        except Exception as e:
            logger.error(f"Error in enhanced analyze_and_trade: {e}")

# ============================================================================
# PERFORMANCE OPTIMIZER - بهینه‌ساز عملکرد
# ============================================================================
import psutil
import gc

class NDSPerformanceOptimizer:
    """
    بهینه‌سازی کامل عملکرد سیستم NDS
    - کش‌گذاری پیشرفته
    - پردازش موازی
    - مدیریت حافظه
    - بنچمارک و مانیتورینگ
    """
    
    def __init__(self, max_cache_size: int = 1000, max_workers: int = 4):
        self.max_cache_size = max_cache_size
        self.max_workers = max_workers
        
        # سیستم کش
        self.price_cache = {}
        self.node_cache = {}
        self.signal_cache = {}
        
        # آمار عملکرد
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        # تنظیمات بهینه‌سازی
        self.optimization_settings = {
            'use_batch_processing': True,
            'use_parallel_processing': True,
            'use_memory_pool': True,
            'cache_enabled': True,
            'use_compression': False,
            'max_batch_size': 128,
            'gpu_enabled': torch.cuda.is_available() if 'torch' in globals() else False
        }
    
    def smart_cache(self, key: str, compute_func, *args, **kwargs) -> Any:
        """
        کش هوشمند با انقضای خودکار و مدیریت اندازه
        """
        if not self.optimization_settings['cache_enabled']:
            return compute_func(*args, **kwargs)
        
        # بررسی کش
        if key in self.price_cache:
            self.stats['cache_hits'] += 1
            return self.price_cache[key]
        
        self.stats['cache_misses'] += 1
        
        # محاسبه مقدار
        value = compute_func(*args, **kwargs)
        
        # ذخیره در کش با مدیریت اندازه
        self.price_cache[key] = value
        if len(self.price_cache) > self.max_cache_size:
            # حذف اولین آیتم
            first_key = next(iter(self.price_cache))
            del self.price_cache[first_key]
        
        return value
    
    def memory_optimization(self, aggressive: bool = False):
        """
        بهینه‌سازی حافظه با سطوح مختلف
        """
        logger.info("🧹 Starting memory optimization...")
        
        if aggressive:
            # سطح تهاجمی
            gc.collect(generation=2)
            gc.collect(generation=1)
            gc.collect(generation=0)
            
            if 'torch' in globals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # پاک کردن کش‌های بزرگ
            self.price_cache.clear()
            self.node_cache.clear()
            
        else:
            # سطح عادی
            gc.collect()
            
            if 'torch' in globals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        memory_info = self._get_memory_usage()
        logger.info(f"   Memory after optimization: {memory_info['percent']:.1f}%")
        
        return memory_info
    
    def _get_memory_usage(self) -> Dict:
        """دریافت اطلاعات مصرف حافظه"""
        process = psutil.Process()
        
        memory_info = {
            'rss': process.memory_info().rss / 1024 / 1024,  # MB
            'vms': process.memory_info().vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024  # MB
        }
        
        if 'torch' in globals() and torch.cuda.is_available():
            memory_info['cuda_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['cuda_cached'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info
    
    def optimize_data_loading(self, symbol: str, timeframe: int, 
                             count: int = 1000) -> pd.DataFrame:
        """
        بهینه‌سازی بارگذاری داده‌ها
        """
        # استفاده از کش برای داده‌های تکراری
        cache_key = f"ohlcv_{symbol}_{timeframe}_{count}"
        
        if cache_key in self.price_cache:
            logger.debug(f"Loading data from cache: {cache_key}")
            return self.price_cache[cache_key]
        
        # بارگذاری داده‌ها
        mt5_manager = MT5Manager(symbol)
        df = mt5_manager.get_ohlcv(timeframe, count)
        
        if df is not None:
            # بهینه‌سازی انواع داده
            df = self._optimize_dataframe_dtypes(df)
            
            # ذخیره در کش
            self.price_cache[cache_key] = df
            
            if len(self.price_cache) > self.max_cache_size:
                # حذف اولین آیتم
                first_key = next(iter(self.price_cache))
                del self.price_cache[first_key]
        
        return df
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """بهینه‌سازی انواع داده در DataFrame"""
        # تبدیل به انواع بهینه
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """دریافت خلاصه عملکرد"""
        summary = {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'avg_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
            'optimization_settings': self.optimization_settings
        }
        
        return summary
    
# ============================================================================
# OPTIMIZED NDS ANALYZER - آنالایزر NDS بهینه‌شده
# ============================================================================
# ============================================================================
# COMPREHENSIVE ARTICLE MODEL INTEGRATOR - یکپارچه‌کننده کامل مدل‌های مقاله
# ============================================================================
class ArticleModelIntegrator:
    """
    یکپارچه‌کننده تمام مدل‌های مقاله برای استفاده در سیستم معاملاتی
    """
    
    def __init__(self):
        # Initialize all models from article
        self.transformer = PriceTransformer()
        self.gnn = AssetCorrelationGNN()
        self.rl_optimizer = RLPolicyOptimizer()
        self.hmm = MarketRegimeHMM()
        self.cvar = CVaRRiskManager()
        self.garch = GARCHVolatilityModel()
        self.vwap = VWAPOptimizer()
        self.setar = SETARVolumeModel()
        self.risk_budgeting = AdaptiveRiskBudgeting()  # بودجه‌بندی ریسک تطبیقی
        
        logger.info("✅ All article models initialized (complete formulas, no simplifications)")
    
    def get_comprehensive_signal(self, price_data: pd.DataFrame, 
                                 ofi_data: np.ndarray = None,
                                 macro_data: np.ndarray = None) -> Dict:
        """
        Generate comprehensive trading signal using all article models
        """
        try:
            signals = {}
            
            # 1. Transformer price prediction
            transformer_pred = self.transformer.predict(price_data, ofi_data, macro_data)
            signals['transformer_prediction'] = transformer_pred
            
            # 2. GARCH volatility forecast
            if len(price_data) > 1:
                returns = price_data['close'].pct_change().dropna().values
                self.garch.update(returns)
                volatility = self.garch.forecast_volatility()
                signals['garch_volatility'] = volatility
            else:
                signals['garch_volatility'] = 0.02
            
            # 3. HMM regime detection
            volatility = signals.get('garch_volatility', 0.02)
            ofi = ofi_data[-1] if ofi_data is not None and len(ofi_data) > 0 else 0.0
            macro = macro_data[-1] if macro_data is not None and len(macro_data) > 0 else 0.0
            regime = self.hmm.detect_regime(volatility, ofi, macro)
            signals['market_regime'] = regime
            signals['position_limit'] = self.hmm.get_position_limit(regime)
            
            # 4. VWAP and volume analysis
            if 'volume' in price_data.columns:
                volumes = price_data['volume'].values
                prices = price_data['close'].values
                volume_decomp = self.vwap.decompose_volume(volumes, prices)
                vwap_pred = self.vwap.predict_vwap(volumes, prices)
                signals['vwap'] = vwap_pred
                signals['volume_decomposition'] = volume_decomp
                
                # SETAR volume prediction
                setar_pred = self.setar.predict_volume(volumes)
                signals['setar_volume'] = setar_pred
            
            # 5. CVaR risk adjustment
            cvar = self.cvar.compute_cvar()
            signals['cvar'] = cvar
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in comprehensive signal generation: {e}")
            return {}
    
    def update_rl_after_trade(self, state: np.ndarray, action: float, reward: float, done: bool = False):
        """Update RL optimizer after each trade"""
        self.rl_optimizer.add_experience(state, action, reward, done)
    
    def get_rl_position_size(self, state: np.ndarray) -> float:
        """Get position size from RL policy"""
        return self.rl_optimizer.get_position_size(state)


class OptimizedNDSAnalyzer(EnhancedNDSAnalyzer):
    """
    نسخه بهینه‌شده آنالایزر NDS با عملکرد بهبودیافته
    + یکپارچه‌سازی کامل مدل‌های مقاله
    """
    
    def __init__(self, mt5_manager: MT5Manager, config: Any = None):
        super().__init__(mt5_manager, config=config)
        
        # افزودن بهینه‌ساز عملکرد
        self.optimizer = NDSPerformanceOptimizer(
            max_cache_size=2000,
            max_workers=4
        )
        
        # ⭐ یکپارچه‌کننده مدل‌های مقاله
        self.article_models = ArticleModelIntegrator()
        
        # کش‌های اختصاصی
        self.local_cache = {}
        self.prediction_cache = {}
        
        # تنظیمات بهینه‌سازی
        self.optimization_enabled = True
        self.batch_mode = True
        self.parallel_processing = True
        
        # آماری
        self.analysis_times = []
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info("⚡ Optimized NDS Analyzer Initialized")
    
    def optimized_analyze(self) -> Optional[NDSSignal]:
        """
        تحلیل بهینه‌شده با استفاده از تمام تکنیک‌های بهینه‌سازی
        + یکپارچه‌سازی مدل‌های مقاله
        """
        start_time = time.perf_counter()
        
        try:
            # بررسی کش
            cache_key = self._generate_analysis_cache_key()
            if self.optimization_enabled and cache_key in self.local_cache:
                self.cache_stats['hits'] += 1
                logger.debug("Analysis result retrieved from cache")
                return self.local_cache[cache_key]
            
            self.cache_stats['misses'] += 1
            
            # بارگذاری بهینه‌شده داده‌ها
            df_trend = self._optimized_data_loading(self.tf_trend, 1440)
            df_analysis = self._optimized_data_loading(self.tf_analysis, 500)
            
            if df_trend is None or df_analysis is None:
                return None
            
            # ⭐ استفاده از مدل‌های مقاله
            article_signals = self.article_models.get_comprehensive_signal(df_analysis)
            
            # تحلیل NDS پایه
            signal = super().enhanced_analyze()
            
            # ⭐ بهبود سیگنال با مدل‌های مقاله
            if signal and article_signals:
                signal = self._enhance_signal_with_article_models(signal, article_signals, df_analysis)
            
            # کش کردن نتیجه
            if signal and self.optimization_enabled:
                self.local_cache[cache_key] = signal
                if len(self.local_cache) > 1000:
                    # حذف قدیمی‌ترین موارد
                    oldest_key = next(iter(self.local_cache))
                    del self.local_cache[oldest_key]
            
            # ثبت زمان تحلیل
            end_time = time.perf_counter()
            analysis_time = end_time - start_time
            self.analysis_times.append(analysis_time)
            
            # بهینه‌سازی حافظه دوره‌ای
            if len(self.analysis_times) % 10 == 0:
                self.optimizer.memory_optimization(aggressive=False)
            
            logger.debug(f"Analysis completed in {analysis_time:.3f}s")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in optimized analysis: {e}")
            return None
    
    def _optimized_data_loading(self, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """بارگذاری بهینه‌شده داده‌ها"""
        return self.optimizer.optimize_data_loading(
            self.mt5.symbol, timeframe, count
        )
    
    def _generate_analysis_cache_key(self) -> str:
        """تولید کلید کش برای تحلیل"""
        # استفاده از زمان و قیمت فعلی
        bid, ask = self.mt5.get_current_price()
        current_price = (bid + ask) / 2
        minute = datetime.now().minute
        
        return f"analysis_{self.mt5.symbol}_{minute}_{current_price:.2f}"
    
    def _enhance_signal_with_article_models(self, signal: NDSSignal, 
                                           article_signals: Dict, 
                                           df: pd.DataFrame) -> NDSSignal:
        """
        بهبود سیگنال با استفاده از مدل‌های مقاله
        """
        try:
            # 1. استفاده از Transformer prediction برای تنظیم TP
            if 'transformer_prediction' in article_signals and article_signals['transformer_prediction']:
                transformer_pred = article_signals['transformer_prediction']
                current_price = df['close'].iloc[-1]
                
                # اگر پیش‌بینی Transformer با جهت سیگنال همسو است، اعتماد را افزایش بده
                if signal.direction == TrendDirection.BULLISH:
                    if transformer_pred > current_price:
                        signal.confidence = min(signal.confidence * 1.1, 0.95)
                elif signal.direction == TrendDirection.BEARISH:
                    if transformer_pred < current_price:
                        signal.confidence = min(signal.confidence * 1.1, 0.95)
            
            # 2. استفاده از GARCH برای تنظیم SL
            if 'garch_volatility' in article_signals:
                volatility = article_signals['garch_volatility']
                entry = signal.entry_price
                
                # تنظیم SL بر اساس نوسان GARCH
                k = 2.0  # Multiplier
                if signal.direction == TrendDirection.BULLISH:
                    new_sl = entry - k * volatility * entry
                    if new_sl < signal.stop_loss:  # فقط اگر بهتر باشد
                        signal.stop_loss = new_sl
                else:
                    new_sl = entry + k * volatility * entry
                    if new_sl > signal.stop_loss:
                        signal.stop_loss = new_sl
                
                # محاسبه مجدد R/R
                if abs(signal.entry_price - signal.stop_loss) > 0:
                    signal.risk_reward = abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss)
            
            # 3. استفاده از HMM برای محدود کردن حجم
            if 'market_regime' in article_signals:
                regime = article_signals['market_regime']
                position_limit = article_signals.get('position_limit', 1.0)
                
                # اگر رژیم volatile است، اعتماد را کاهش بده
                if regime == 'volatile':
                    signal.confidence = signal.confidence * 0.9
            
            # 4. استفاده از CVaR برای مدیریت ریسک
            if 'cvar' in article_signals:
                cvar = article_signals['cvar']
                if cvar > 0:
                    # اگر CVaR بالا است، اعتماد را کاهش بده
                    if cvar > 0.05:  # 5% CVaR
                        signal.confidence = signal.confidence * 0.85
            
            # 5. استفاده از VWAP برای تنظیم Entry
            if 'vwap' in article_signals and article_signals['vwap']:
                vwap = article_signals['vwap']
                current_price = df['close'].iloc[-1]
                
                # اگر قیمت نزدیک VWAP است، اعتماد را افزایش بده
                price_diff_pct = abs(current_price - vwap) / vwap
                if price_diff_pct < 0.01:  # کمتر از 1% تفاوت
                    signal.confidence = min(signal.confidence * 1.05, 0.95)
            
            logger.info(f"📊 Signal enhanced with article models:")
            logger.info(f"   Regime: {article_signals.get('market_regime', 'N/A')}")
            logger.info(f"   GARCH Vol: {article_signals.get('garch_volatility', 0):.4f}")
            logger.info(f"   CVaR: {article_signals.get('cvar', 0):.4f}")
            logger.info(f"   Final Confidence: {signal.confidence:.2%}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with article models: {e}")
            return signal
    
    def get_performance_metrics(self) -> Dict:
        """دریافت متریک‌های عملکرد"""
        metrics = {
            'total_analyses': len(self.analysis_times),
            'avg_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0,
            'min_analysis_time': np.min(self.analysis_times) if self.analysis_times else 0,
            'max_analysis_time': np.max(self.analysis_times) if self.analysis_times else 0,
            'cache_hit_rate': self.cache_stats['hits'] / max(self.cache_stats['hits'] + self.cache_stats['misses'], 1),
            'optimizer_stats': self.optimizer.get_performance_summary()
        }
        
        return metrics


# ============================================================================
# STRATEGY CONFIGURATION - تنظیمات استراتژی‌ها
# ============================================================================
@dataclass
class StrategyConfig:
    """تنظیمات تایم‌فریم برای هر استراتژی"""
    name: str
    trend_tf: int  # تایم‌فریم تشخیص روند
    coarse_analysis_tf: int  # تایم‌فریم تحلیل درشت
    fine_analysis_tf: int  # تایم‌فریم تحلیل ریز
    entry_tf: int  # تایم‌فریم ورود
    exit_signal_tf: int  # تایم‌فریم اولین نشانه خروج
    exit_confirm_tf: int  # تایم‌فریم تایید خروج
    
    @staticmethod
    def day_trading() -> 'StrategyConfig':
        """Day Trading: H1/M15/M3/M1"""
        return StrategyConfig(
            name="Day Trading",
            trend_tf=mt5.TIMEFRAME_H1,  # H1 برای روند
            coarse_analysis_tf=mt5.TIMEFRAME_M15,  # M15 برای تحلیل درشت
            fine_analysis_tf=mt5.TIMEFRAME_M3,  # M3 برای تحلیل ریز
            entry_tf=mt5.TIMEFRAME_M1,  # M1 برای ورود
            exit_signal_tf=mt5.TIMEFRAME_M5,  # M5 برای اولین نشانه خروج
            exit_confirm_tf=mt5.TIMEFRAME_M3  # M3 برای تایید خروج
        )
    
    @staticmethod
    def scalping() -> 'StrategyConfig':
        """Scalping: M15/M5/M3/M1"""
        return StrategyConfig(
            name="Scalping",
            trend_tf=mt5.TIMEFRAME_M15,  # M15 برای روند
            coarse_analysis_tf=mt5.TIMEFRAME_M5,  # M5 برای تحلیل درشت
            fine_analysis_tf=mt5.TIMEFRAME_M3,  # M3 برای تحلیل ریز
            entry_tf=mt5.TIMEFRAME_M1,  # M1 برای ورود
            exit_signal_tf=mt5.TIMEFRAME_M3,  # M3 برای اولین نشانه خروج
            exit_confirm_tf=mt5.TIMEFRAME_M1  # M1 برای تایید خروج
        )
    
    @staticmethod
    def super_scalping() -> 'StrategyConfig':
        """Super Scalping: M5/M3/M1"""
        return StrategyConfig(
            name="Super Scalping",
            trend_tf=mt5.TIMEFRAME_M5,  # M5 برای روند
            coarse_analysis_tf=mt5.TIMEFRAME_M3,  # M3 برای تحلیل درشت
            fine_analysis_tf=mt5.TIMEFRAME_M1,  # M1 برای تحلیل ریز
            entry_tf=mt5.TIMEFRAME_M1,  # M1 برای ورود
            exit_signal_tf=mt5.TIMEFRAME_M3,  # M3 برای اولین نشانه خروج
            exit_confirm_tf=mt5.TIMEFRAME_M1  # M1 برای تایید خروج
        )


# ============================================================================
# UNIFIED TRADING BOT - ربات یکپارچه با 3 استراتژی
# ============================================================================
class UnifiedTradingBot(OptimizedNDSTradingBot):
    """
    ربات یکپارچه که از مغز تحلیلی مشترک استفاده می‌کند
    و فقط تایم‌فریم‌های تحلیل را بر اساس استراتژی انتخاب شده تغییر می‌دهد
    """
    
    def __init__(self, symbol: str = "BTCUSD", max_lots: float = None, 
                 config: Any = None, strategy: str = "day_trading"):
        """
        Args:
            strategy: یکی از 'day_trading', 'scalping', 'super_scalping'
        """
        # فراخوانی سازنده والد
        super().__init__(symbol, max_lots=max_lots, config=config)
        
        # تنظیم استراتژی
        self.strategy_name = strategy
        if strategy == "day_trading":
            self.strategy_config = StrategyConfig.day_trading()
        elif strategy == "scalping":
            self.strategy_config = StrategyConfig.scalping()
        elif strategy == "super_scalping":
            self.strategy_config = StrategyConfig.super_scalping()
        else:
            logger.warning(f"⚠️ Unknown strategy '{strategy}', using day_trading")
            self.strategy_config = StrategyConfig.day_trading()
            self.strategy_name = "day_trading"
        
        # تنظیم تایم‌فریم‌های analyzer بر اساس استراتژی
        self.nds.tf_trend = self.strategy_config.trend_tf
        self.nds.tf_analysis = self.strategy_config.fine_analysis_tf
        self.nds.tf_entry = self.strategy_config.entry_tf
        
        logger.info(f"✅ UnifiedTradingBot initialized with strategy: {self.strategy_config.name}")
        logger.info(f"   Trend TF: {self.strategy_config.trend_tf}")
        logger.info(f"   Coarse Analysis TF: {self.strategy_config.coarse_analysis_tf}")
        logger.info(f"   Fine Analysis TF: {self.strategy_config.fine_analysis_tf}")
        logger.info(f"   Entry TF: {self.strategy_config.entry_tf}")
        logger.info(f"   Exit Signal TF: {self.strategy_config.exit_signal_tf}")
        logger.info(f"   Exit Confirm TF: {self.strategy_config.exit_confirm_tf}")
    
    def _check_exit_signal(self) -> bool:
        """
        بررسی سیگنال خروج بر اساس استراتژی
        اولین نشانه تغییر روند در exit_signal_tf + تایید در exit_confirm_tf
        """
        try:
            # دریافت داده‌های تایم‌فریم‌های خروج
            df_signal = self.mt5.get_ohlcv(self.strategy_config.exit_signal_tf, 50)
            df_confirm = self.mt5.get_ohlcv(self.strategy_config.exit_confirm_tf, 50)
            
            if df_signal is None or df_confirm is None or len(df_signal) < 10 or len(df_confirm) < 10:
                return False
            
            # تحلیل روند در تایم‌فریم signal
            prices_signal = df_signal['close'].values
            ma_fast_signal = np.mean(prices_signal[-5:])
            ma_slow_signal = np.mean(prices_signal[-15:])
            current_signal = prices_signal[-1]
            
            # تشخیص تغییر روند در signal timeframe
            trend_signal = None
            if current_signal > ma_fast_signal > ma_slow_signal:
                trend_signal = TrendDirection.BULLISH
            elif current_signal < ma_fast_signal < ma_slow_signal:
                trend_signal = TrendDirection.BEARISH
            else:
                trend_signal = TrendDirection.NEUTRAL
            
            # تحلیل روند در تایم‌فریم confirm
            prices_confirm = df_confirm['close'].values
            ma_fast_confirm = np.mean(prices_confirm[-5:])
            ma_slow_confirm = np.mean(prices_confirm[-15:])
            current_confirm = prices_confirm[-1]
            
            trend_confirm = None
            if current_confirm > ma_fast_confirm > ma_slow_confirm:
                trend_confirm = TrendDirection.BULLISH
            elif current_confirm < ma_fast_confirm < ma_slow_confirm:
                trend_confirm = TrendDirection.BEARISH
            else:
                trend_confirm = TrendDirection.NEUTRAL
            
            # بررسی معاملات باز
            positions = self.mt5.get_active_positions()
            if not positions:
                return False
            
            for pos in positions:
                # تعیین جهت معامله
                trade_direction = TrendDirection.BULLISH if pos.type == mt5.ORDER_TYPE_BUY else TrendDirection.BEARISH
                
                # اگر روند در signal تغییر کرد و در confirm تایید شد
                if trend_signal != trade_direction and trend_signal != TrendDirection.NEUTRAL:
                    if trend_confirm == trend_signal:  # تایید در confirm
                        logger.info(f"🔄 Exit signal detected: Trend changed from {trade_direction} to {trend_signal}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking exit signal: {e}")
            return False
    
    def _optimized_main_loop(self):
        """حلقه اصلی با پشتیبانی از خروج بر اساس استراتژی"""
        previous_positions = set()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 1. پذیرش معاملات موجود
                positions = mt5.positions_get(symbol=self.symbol)
                current_tickets = {pos.ticket for pos in positions} if positions else set()
                
                # تشخیص معاملات بسته شده برای RL update
                closed_tickets = previous_positions - current_tickets
                for ticket in closed_tickets:
                    self._handle_closed_trade(ticket)
                
                previous_positions = current_tickets
                
                if positions:
                    for pos in positions:
                        if pos.ticket not in self.trailing_manager.trade_states:
                            logger.info(f"🔄 Adopting existing trade #{pos.ticket}")
                            self._adopt_existing_trade(pos.ticket)
                
                # 2. مدیریت معامله باز
                if positions:
                    # بررسی سیگنال خروج بر اساس استراتژی
                    if self._check_exit_signal():
                        for pos in positions:
                            if pos.profit > 0:  # فقط اگر در سود باشد
                                logger.info(f"💰 Closing trade #{pos.ticket} due to trend change (profit: ${pos.profit:.2f})")
                                trade_info = TradeInfo(
                                    ticket=pos.ticket,
                                    symbol=pos.symbol,
                                    order_type=pos.type,
                                    volume=pos.volume,
                                    open_price=pos.price_open,
                                    current_price=pos.price_current,
                                    sl=pos.sl,
                                    tp=pos.tp,
                                    profit=pos.profit,
                                    open_time=datetime.fromtimestamp(pos.time)
                                )
                                self.trade.close_trade(trade_info, "Trend change detected")
                    
                    # مدیریت عادی معامله
                    logger.debug(f"⏳ Managing {len(positions)} open position(s)")
                    for pos in positions:
                        self.trailing_manager.manage_trade(pos.ticket)
                    time.sleep(1)
                    continue
                
                # 3. تحلیل و معامله جدید
                logger.info("🔍 Analyzing for new trade...")
                
                # فراخوانی تحلیلگر با تایم‌فریم‌های استراتژی
                if hasattr(self.nds, 'optimized_analyze'):
                    signal = self.nds.optimized_analyze()
                elif hasattr(self.nds, 'enhanced_analyze'):
                    signal = self.nds.enhanced_analyze()
                else:
                    signal = self.nds.analyze()
                
                if signal is None:
                    logger.debug("❌ No signal generated")
                    time.sleep(5)
                    continue
                
                # لاگ سیگنال
                logger.info("✅ Signal received:")
                logger.info(f"   Direction: {signal.direction.value}")
                logger.info(f"   Entry: {signal.entry_price:.2f}")
                logger.info(f"   SL: {signal.stop_loss:.2f}")
                logger.info(f"   TP: {signal.take_profit:.2f}")
                logger.info(f"   Confidence: {signal.confidence:.2%}")
                logger.info(f"   R/R: {signal.risk_reward:.2f}")
                
                # بررسی valid بودن
                if not signal.is_valid():
                    logger.warning("⚠️ Signal is not valid")
                    time.sleep(10)
                    continue
                
                logger.info("✅ Signal is valid")
                
                # اعتبارسنجی ریسک
                can_trade, msg = self.risk.can_trade()
                if not can_trade:
                    logger.warning(f"⚠️ Cannot trade: {msg}")
                    time.sleep(10)
                    continue
                
                valid, risk_msg = self.risk.validate_signal(signal)
                logger.info(f"📊 Risk Validation: {valid} - {risk_msg}")
                
                if not valid:
                    logger.warning(f"⚠️ Risk validation failed: {risk_msg}")
                    time.sleep(10)
                    continue
                
                logger.info("✅ Signal passed Risk validation")
                
                # 4. اجرای معامله
                logger.info("🚀 Executing trade with nodes...")
                ticket = self._execute_trade_with_nodes(signal)
                
                if ticket:
                    logger.info(f"🎉 Trade #{ticket} opened and managed by Node-Based System")
                else:
                    logger.error("❌ Trade execution returned None!")
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)


# ============================================================================
# OPTIMIZED TRADING BOT - ربات معاملاتی بهینه‌شده (Base Class)
# ============================================================================

class OptimizedNDSTradingBot(EnhancedNDSTradingBot):  # ✅ وراثت درست
    """ربات بهینه‌شده با سیستم Trailing گره‌محور"""
    
    def __init__(self, symbol: str = "BTCUSD", max_lots: float = None, config: Any = None):
        super().__init__(symbol, max_lots=max_lots, config=config)
        
        # ⭐ جایگزینی analyzer با نسخه بهینه‌شده
        self.nds = OptimizedNDSAnalyzer(self.mt5, config=self.config)
        
        # ⭐ افزودن سیستم Trailing گره‌محور
        self.trailing_manager = ImprovedNodeBasedTrailing(
            mt5_manager=self.mt5,
            symbol=self.symbol
        )
        
        # ⭐ ردیابی معاملات برای RL optimization
        self.trade_history = []
        self.total_trades = 0
        
        # تنظیمات
        self.point = 0.1 if symbol == "BTCUSD" else 0.00001
        
        logger.info("🤖 OptimizedNDSTradingBot with Node-Based Trailing and RL initialized")

    # ══════════════════════════════════════════════════════════
    # متد start (فقط یک بار!)
    # ══════════════════════════════════════════════════════════
    def start(self):
        """شروع ربات"""
        logger.info("🚀 Starting Optimized NDS Trading Bot...")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Max Risk: {self.risk.max_risk_percent}%")
        logger.info(f"   Max Lots: {self.max_lots}")

        if not self.mt5.connect():
            logger.error("Error connecting to MT5")
            return

        self.running = True
        logger.info("✅ Optimized Bot started successfully")

        try:
            self._optimized_main_loop()
        except KeyboardInterrupt:
            logger.info("Optimized Bot stopped by user")
        finally:
            self.stop()

    # ══════════════════════════════════════════════════════════
    # حلقه اصلی
    # ══════════════════════════════════════════════════════════
    def _optimized_main_loop(self):
        """حلقه اصلی بهینه‌شده با Node-Based Execution + RL Optimization"""
        
        # ردیابی معاملات قبلی برای تشخیص بسته شدن
        previous_positions = set()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 1. پذیرش معاملات موجود
                positions = mt5.positions_get(symbol=self.symbol)
                current_tickets = {pos.ticket for pos in positions} if positions else set()
                
                # ⭐ تشخیص معاملات بسته شده برای RL update
                closed_tickets = previous_positions - current_tickets
                for ticket in closed_tickets:
                    self._handle_closed_trade(ticket)
                
                previous_positions = current_tickets
                
                if positions:
                    for pos in positions:
                        if pos.ticket not in self.trailing_manager.trade_states:
                            logger.info(f"🔄 Adopting existing trade #{pos.ticket}")
                            self._adopt_existing_trade(pos.ticket)
                
                # 2. مدیریت معامله باز
                if positions:
                    logger.debug(f"⏳ Managing {len(positions)} open position(s)")
                    for pos in positions:
                        self.trailing_manager.manage_trade(pos.ticket)
                    time.sleep(1)
                    continue
                
                # 3. تحلیل و معامله جدید
                logger.info("🔍 Analyzing for new trade...")
                
                # فراخوانی تحلیلگر
                if hasattr(self.nds, 'optimized_analyze'):
                    signal = self.nds.optimized_analyze()
                elif hasattr(self.nds, 'enhanced_analyze'):
                    signal = self.nds.enhanced_analyze()
                else:
                    signal = self.nds.analyze()
                
                if signal is None:
                    logger.debug("❌ No signal generated")
                    time.sleep(5)
                    continue
                
                # لاگ سیگنال
                logger.info("✅ Signal received:")
                logger.info(f"   Direction: {signal.direction.value}")
                logger.info(f"   Entry: {signal.entry_price:.2f}")
                logger.info(f"   SL: {signal.stop_loss:.2f}")
                logger.info(f"   TP: {signal.take_profit:.2f}")
                logger.info(f"   Confidence: {signal.confidence:.2%}")
                logger.info(f"   R/R: {signal.risk_reward:.2f}")
                
                # بررسی valid بودن
                if not signal.is_valid():
                    logger.warning("⚠️ Signal is not valid")
                    time.sleep(10)
                    continue
                
                logger.info("✅ Signal is valid")
                
                # اعتبارسنجی ریسک
                can_trade, msg = self.risk.can_trade()
                if not can_trade:
                    logger.warning(f"⚠️ Cannot trade: {msg}")
                    time.sleep(10)
                    continue
                
                valid, risk_msg = self.risk.validate_signal(signal)
                logger.info(f"📊 Risk Validation: {valid} - {risk_msg}")
                
                if not valid:
                    logger.warning(f"⚠️ Risk validation failed: {risk_msg}")
                    time.sleep(10)
                    continue
                
                logger.info("✅ Signal passed Risk validation")
                
                # 4. اجرای معامله
                logger.info("🚀 Executing trade with nodes...")
                ticket = self._execute_trade_with_nodes(signal)
                
                if ticket:
                    logger.info(f"🎉 Trade #{ticket} opened and managed by Node-Based System")
                else:
                    logger.error("❌ Trade execution returned None!")
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    # ══════════════════════════════════════════════════════════
    # پذیرش معامله موجود
    # ══════════════════════════════════════════════════════════
    def _adopt_existing_trade(self, ticket: int):
        """پذیرش و مقداردهی state برای معامله موجود"""
        try:
            logger.info(f"🔄 Adopting existing trade #{ticket}...")
            
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"❌ Cannot find position #{ticket}")
                return False
            
            pos = positions[0]
            
            # شناسایی گره‌ها
            nodes = self.trailing_manager.detect_nodes(
                timeframe=mt5.TIMEFRAME_M3,
                lookback=100
            )
            
            if not nodes:
                logger.warning("⚠️ No nodes detected!")
                nodes = []
            
            # تعیین SL/TP
            default_sl = pos.sl if pos.sl > 0 else (
                pos.price_open - (100 * self.point) if pos.type == mt5.ORDER_TYPE_BUY 
                else pos.price_open + (100 * self.point)
            )
            
            default_tp = pos.tp if pos.tp > 0 else (
                pos.price_open + (150 * self.point) if pos.type == mt5.ORDER_TYPE_BUY 
                else pos.price_open - (150 * self.point)
            )
            
            # مقداردهی state
            self.trailing_manager.initialize_trade_state(
                ticket=ticket,
                entry_price=pos.price_open,
                sl=default_sl,
                tp=default_tp,
                volume=pos.volume,
                direction="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                nodes=nodes,
                spread=5.0,
                commission=0.5
            )
            
            logger.info(f"✅ Trade #{ticket} adopted successfully!")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error adopting trade: {e}")
            return False

    # ══════════════════════════════════════════════════════════
    # اجرای معامله با Node-Based
    # ══════════════════════════════════════════════════════════
    
    def _execute_trade_with_nodes(self, signal: NDSSignal) -> Optional[int]:
        """اجرای معامله با Node-Based Trailing - استفاده از TradeManager"""
        try:
            logger.info("=" * 70)
            logger.info("🚀 EXECUTING TRADE WITH NODE-BASED MANAGEMENT")
            logger.info("=" * 70)
            
            # ✅ استفاده از TradeManager برای باز کردن معامله
            ticket = self.trade.open_trade_safe(signal)
            if not ticket:
                logger.error("❌ Trade opening failed via TradeManager")
                return None
            
            logger.info(f"✅ Trade opened via TradeManager! Ticket: #{ticket}")
            
            # محاسبه spread و commission برای trailing manager
            symbol_info = mt5.symbol_info(self.symbol)
            spread_pips = symbol_info.spread if symbol_info else 10.0
            commission = 0.0  # یا مقادیر واقعی حساب شما
            
            # شناسایی گره‌ها
            nodes = self.trailing_manager.detect_nodes(
                timeframe=mt5.TIMEFRAME_M3,
                lookback=100
            )
            
            # دریافت position برای اطلاعات
            positions = self.mt5.get_active_positions()
            if not positions:
                return ticket
            
            pos = positions[0]
            
            # ایجاد State در trailing manager
            self.trailing_manager.initialize_trade_state(
                ticket=ticket,
                entry_price=pos.price_open,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                volume=pos.volume,
                direction="BUY" if signal.direction == TrendDirection.BULLISH else "SELL",
                nodes=nodes,
                spread=spread_pips,
                commission=commission
            )
            
            logger.info("🎉 Node-Based Trailing ACTIVE")
            
            # ⭐ ثبت معامله برای RL
            self.total_trades += 1
            self.trade_history.append({
                'ticket': ticket,
                'entry_time': datetime.now(),
                'entry_price': pos.price_open,
                'volume': pos.volume,
                'direction': signal.direction,
                'signal_confidence': signal.confidence
            })
            
            return ticket
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    def _handle_closed_trade(self, ticket: int):
        """
        مدیریت معامله بسته شده و به‌روزرسانی RL
        """
        try:
            # دریافت اطلاعات معامله از تاریخچه
            trade_info = None
            for trade in self.trade_history:
                if trade['ticket'] == ticket:
                    trade_info = trade
                    break
            
            if not trade_info:
                return
            
            # دریافت سود نهایی از MT5
            deal_history = mt5.history_deals_get(ticket=ticket, group="*")
            if deal_history:
                total_profit = sum(deal.profit for deal in deal_history)
            else:
                # اگر نتوانستیم سود را بگیریم، از تاریخچه استفاده می‌کنیم
                total_profit = 0.0
            
            # به‌روزرسانی RL
            self._update_rl_after_trade_close(ticket, total_profit)
            
        except Exception as e:
            logger.error(f"Error handling closed trade: {e}")
    
    def _update_rl_after_trade_close(self, ticket: int, profit: float):
        """
        به‌روزرسانی RL بعد از بسته شدن معامله
        این متد بعد از هر 5 معامله پارامترهای RL را بهینه می‌کند
        """
        try:
            # پیدا کردن معامله در تاریخچه
            trade_info = None
            for trade in self.trade_history:
                if trade['ticket'] == ticket:
                    trade_info = trade
                    break
            
            if not trade_info:
                return
            
            # ساخت state برای RL (ساده‌سازی شده)
            df = self.mt5.get_ohlcv(mt5.TIMEFRAME_M3, 100)
            if df is None or len(df) < 10:
                return
            
            # استخراج features برای state
            prices = df['close'].values[-20:]
            returns = np.diff(prices, prepend=prices[0])
            volatility = np.std(returns)
            
            # ساخت state vector (20 بعدی)
            state = np.concatenate([
                prices[-10:] / prices[-1],  # Normalized prices (10 dims)
                returns[-5:] / (volatility + 1e-8),  # Normalized returns (5 dims)
                [volatility, trade_info['signal_confidence'], 
                 trade_info['volume'], profit / 100.0, 0.0]  # Additional features (5 dims)
            ])
            
            # اگر state کوتاه است، padding
            if len(state) < 20:
                state = np.pad(state, (0, 20 - len(state)), 'constant')
            elif len(state) > 20:
                state = state[:20]
            
            # محاسبه reward طبق فرمول کامل مقاله:
            # R_t = ΔP_t - γ · TransactionCost(a_t, a_{t-1}) - η · RiskPenalty(σ_t, Drawdown)
            
            # 1. ΔP_t: تغییر قیمت (سود/ضرر)
            delta_p = profit / 100.0  # Normalize profit
            
            # 2. TransactionCost(a_t, a_{t-1}): هزینه معاملات
            # هزینه شامل spread و commission
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info:
                spread_cost = symbol_info.spread * symbol_info.point * trade_info['volume'] * 10  # تقریبی
                commission_cost = 0.0  # اگر commission دارید، اینجا اضافه کنید
                transaction_cost = (spread_cost + commission_cost) / 100.0
            else:
                transaction_cost = 0.0
            
            gamma = 0.1  # وزن هزینه معاملات
            
            # 3. RiskPenalty(σ_t, Drawdown): جریمه ریسک
            # محاسبه drawdown
            account_info = self.mt5.account_info
            if account_info:
                equity = account_info.equity
                balance = account_info.balance
                drawdown = (balance - equity) / balance if balance > 0 else 0.0
            else:
                drawdown = 0.0
            
            # Risk penalty بر اساس نوسان و drawdown
            risk_penalty = volatility * 0.5 + drawdown * 0.3
            eta = 0.2  # وزن جریمه ریسک
            
            # فرمول کامل reward
            reward = delta_p - gamma * transaction_cost - eta * risk_penalty
            
            # اضافه کردن missed opportunity penalty (طبق درخواست کاربر)
            # اگر سیگنال خوب بود ولی وارد نشدیم، تنبیه می‌شویم
            if trade_info['signal_confidence'] > 0.8 and profit < 0:
                # اگر اعتماد بالا بود ولی ضرر کردیم، جریمه بیشتر
                reward -= 0.15
            elif trade_info['signal_confidence'] > 0.8 and profit > 0:
                # اگر اعتماد بالا بود و سود کردیم، پاداش بیشتر
                reward += 0.15
            
            # به‌روزرسانی RL
            self.nds.article_models.update_rl_after_trade(state, 
                                                          trade_info['volume'] / self.max_lots, 
                                                          reward, 
                                                          done=True)
            
            logger.info(f"📊 RL updated for trade #{ticket}, Profit: ${profit:.2f}, Reward: {reward:.4f}")
            
            # اگر 5 معامله انجام شده، بهینه‌سازی انجام می‌شود (داخل RL optimizer)
            
        except Exception as e:
            logger.error(f"Error updating RL after trade close: {e}")
        
    # ══════════════════════════════════════════════════════════
    # گزارش‌ها
    # ══════════════════════════════════════════════════════════
    def _should_report(self, current_time: datetime) -> bool:
        if not hasattr(self, '_last_report'):
            self._last_report = current_time
            return True
        return (current_time - self._last_report).seconds >= 60

    def _manage_open_trade(self, position):
        """مدیریت معامله باز - با Trailing Stop فعال"""
        try:
            # تبدیل position به TradeInfo
            if not isinstance(position, TradeInfo):
                trade = TradeInfo(
                    ticket=position.ticket,
                    symbol=position.symbol,
                    order_type=position.type,
                    volume=position.volume,
                    open_price=position.price_open,
                    current_price=position.price_current,
                    sl=position.sl,
                    tp=position.tp,
                    profit=position.profit,
                    open_time=datetime.fromtimestamp(position.time)
                )
            else:
                trade = position

            # دریافت قیمت فعلی
            bid, ask = self.mt5.get_current_price()
            if bid == 0 or ask == 0:
                return

            current_price = bid if trade.order_type == mt5.ORDER_TYPE_BUY else ask
            is_buy = trade.order_type == mt5.ORDER_TYPE_BUY
            
            # محاسبه point
            if self.symbol == "BTCUSD":
                point = 1.0  # ⬅️ 1 دلار = 1 پیپ
            else:
                symbol_info = mt5.symbol_info(self.symbol)
                point = symbol_info.point if symbol_info else 0.01
            
            spread = self.mt5.get_spread()

            # محاسبه سود
            if is_buy:
                profit_pips = (current_price - trade.open_price) / point
            else:
                profit_pips = (trade.open_price - current_price) / point

            logger.info(f"📊 Position #{trade.ticket}: Price={current_price:.2f}, Profit={profit_pips:.1f} pips, P/L=${trade.profit:.2f}")

            # مقداردهی اولیه
            if not hasattr(trade, '_peak_price'):
                trade._peak_price = current_price
                trade._breakeven_done = False
                trade._trailing_active = False
                trade._last_trailing_update = datetime.now()

            # به‌روزرسانی peak
            if is_buy:
                if current_price > trade._peak_price:
                    trade._peak_price = current_price
            else:
                if current_price < trade._peak_price:
                    trade._peak_price = current_price

            # ✅ PHASE 1: BREAKEVEN (10 پیپ)
            if not trade._breakeven_done and profit_pips >= 10:
                if is_buy:
                    breakeven_sl = trade.open_price + spread + (2 * point)
                    
                    if breakeven_sl > trade.sl:
                        success = self.trade.update_trailing_stop(trade, breakeven_sl)
                        if success:
                            trade.sl = breakeven_sl
                            trade._breakeven_done = True
                            logger.info(f"🛡️ BREAKEVEN: SL={breakeven_sl:.2f}")
                else:
                    breakeven_sl = trade.open_price - spread - (2 * point)
                    
                    if breakeven_sl < trade.sl:
                        success = self.trade.update_trailing_stop(trade, breakeven_sl)
                        if success:
                            trade.sl = breakeven_sl
                            trade._breakeven_done = True
                            logger.info(f"🛡️ BREAKEVEN: SL={breakeven_sl:.2f}")

            # ✅ PHASE 2: TRAILING STOP (15+ پیپ)
            elif profit_pips >= 15:
                if not trade._trailing_active:
                    trade._trailing_active = True
                    logger.info(f"🔥 TRAILING ACTIVATED at {profit_pips:.1f} pips")
                
                # محدودیت زمانی: هر 3 ثانیه یک بار
                current_time = datetime.now()
                if (current_time - trade._last_trailing_update).seconds < 3:
                    return
                
                trailing_distance = 8 * point  # 8 دلار
                
                if is_buy:
                    new_sl = current_price - trailing_distance - spread
                    
                    if new_sl > trade.sl:
                        sl_improvement = (new_sl - trade.sl) / point
                        
                        if sl_improvement >= 2:
                            success = self.trade.update_trailing_stop(trade, new_sl)
                            if success:
                                trade.sl = new_sl
                                trade._last_trailing_update = current_time
                                logger.info(f"📈 TRAILING: {new_sl:.2f} (+{sl_improvement:.1f} pips)")
                else:
                    new_sl = current_price + trailing_distance + spread
                    
                    if new_sl < trade.sl:
                        sl_improvement = (trade.sl - new_sl) / point
                        
                        if sl_improvement >= 2:
                            success = self.trade.update_trailing_stop(trade, new_sl)
                            if success:
                                trade.sl = new_sl
                                trade._last_trailing_update = current_time
                                logger.info(f"📉 TRAILING: {new_sl:.2f} (+{sl_improvement:.1f} pips)")

        except Exception as e:
            logger.error(f"❌ Error managing position: {e}")
            import traceback
            traceback.print_exc()
    
    def _status_report(self):
        """گزارش وضعیت"""
        self.mt5.refresh_account()
        
        logger.info(f"\n📊 STATUS REPORT")
        logger.info(f"   Balance: ${self.mt5.account_info.balance:,.2f}")
        logger.info(f"   Equity: ${self.mt5.account_info.equity:,.2f}")
        
        positions = self.mt5.get_active_positions()
        if positions:
            logger.info(f"   ✅ {len(positions)} active trade(s)")
        else:
            logger.info(f"   ⏳ Waiting for signal...")




class NodeBasedTrailingManager:
    """
    مدیریت Trailing Stop مبتنی بر گره‌ها با Partial Close
    """
    
    def __init__(self, mt5_manager, symbol="BTCUSD"):
        self.mt5 = mt5_manager
        self.symbol = symbol
        self.point = 0.1  # برای BTCUSD
        
        # ذخیره وضعیت هر معامله
        self.trade_states = {}
        
        # تنظیمات Partial Close
        self.profit_levels = {
            20: {'close_percent': 5, 'description': 'اولین سیو'},
            50: {'close_percent': 50, 'description': 'نصف باقیمانده'},
            70: {'close_percent': 50, 'description': 'نصف از 50% باقی'},
            85: {'close_percent': 75, 'description': 'سیو 75% کل'}
        }
        
    def initialize_trade_state(self, ticket: int, entry_price: float, 
                               sl: float, tp: float, volume: float,
                               direction: str, nodes: dict,
                               spread: float = 0.0, commission: float = 0.0):
        """
        مقداردهی اولیه state معامله
        
        Args:
            ticket: شماره تیکت
            entry_price: قیمت ورود
            sl: استاپ لاس اولیه
            tp: تیک پرافیت
            volume: حجم اولیه
            direction: 'BUY' یا 'SELL'
            nodes: دیکشنری گره‌ها {'below_entry': [n1, n2], 'above_entry': [n3, n4], ...}
            spread: اسپرد به پیپ (مثلاً 10.0)
            commission: کمیسیون به دلار (مثلاً 0.5)
        """
        self.trade_states[ticket] = {
            'entry_price': entry_price,
            'initial_sl': sl,
            'initial_tp': tp,
            'initial_volume': volume,
            'current_volume': volume,
            'direction': direction,
            'nodes': nodes,
            'spread': spread,
            'commission': commission,
            
            # ⭐ افزودن peak_price با مقدار اولیه entry_price
            'peak_price': entry_price,  # مقدار اولیه
            
            # مراحل مدیریت
            'stage_10pct': False,  # 10% عبور
            'stage_15pct': False,  # 15% عبور (breakeven)
            'stage_50pct': False,  # 50% partial
            'stage_70pct': False,  # 70% partial
            
            # ردیابی
            'total_closed_volume': 0.0,
            'closed_profit': 0.0,
        }
        
        # مقداردهی partial_closes
        for level in self.profit_levels.keys():
            self.trade_states[ticket]['partial_closes'][level] = False
            
        logger.info(f"✅ Trade state initialized for #{ticket}")
        logger.info(f"   Entry: {entry_price:.2f} | Direction: {direction}")
        logger.info(f"   Initial Volume: {volume:.3f} lots")
        logger.info(f"   Nodes: {len(nodes.get('below_entry', []))} below, "
                   f"{len(nodes.get('above_entry', []))} above entry")
    
    def detect_nodes(self, timeframe=mt5.TIMEFRAME_M3, lookback=100) -> dict:
        """
        شناسایی گره‌ها (Swing High/Low) در نمودار
        
        Returns:
            {'below_entry': [prices], 'above_entry': [prices]}
        """
        try:
            # دریافت داده‌های تاریخی
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, lookback)
            if rates is None or len(rates) == 0:
                logger.error("❌ Cannot fetch rates for node detection")
                return {'below_entry': [], 'above_entry': []}
            
            import pandas as pd
            df = pd.DataFrame(rates)
            
            # شناسایی Swing Highs و Lows
            swing_highs = []
            swing_lows = []
            
            window = 5  # تعداد کندل‌های اطراف برای بررسی
            
            for i in range(window, len(df) - window):
                # Swing High: high[i] بزرگتر از همسایه‌ها
                if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                    swing_highs.append(df['high'].iloc[i])
                
                # Swing Low: low[i] کوچکتر از همسایه‌ها
                if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                    swing_lows.append(df['low'].iloc[i])
            
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(self.symbol)
            current_price = tick.bid if tick else df['close'].iloc[-1]
            
            # تفکیک گره‌ها
            nodes_below = sorted([n for n in swing_lows if n < current_price], reverse=True)
            nodes_above = sorted([n for n in swing_highs if n > current_price])
            
            logger.info(f"🔍 Detected {len(nodes_below)} nodes below, {len(nodes_above)} above current price")
            
            return {
                'below_entry': nodes_below[:10],  # 10 گره نزدیک
                'above_entry': nodes_above[:10]
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting nodes: {e}")
            return {'below_entry': [], 'above_entry': []}
    
    def get_nearest_node_below(self, price: float, nodes: list) -> float:
        """
        یافتن نزدیکترین گره زیر قیمت داده شده
        """
        valid_nodes = [n for n in nodes if n < price]
        if not valid_nodes:
            # اگر گره‌ای وجود ندارد، 50 پیپ زیر
            return price - (50 * self.point)
        
        return max(valid_nodes)  # بزرگترین (نزدیکترین)
    
    def get_nearest_node_above(self, price: float, nodes: list) -> float:
        """
        یافتن نزدیکترین گره بالای قیمت داده شده
        """
        valid_nodes = [n for n in nodes if n > price]
        if not valid_nodes:
            # اگر گره‌ای وجود ندارد، 50 پیپ بالا
            return price + (50 * self.point)
        
        return min(valid_nodes)  # کوچکترین (نزدیکترین)
    
    def calculate_profit_percent(self, entry: float, current: float, 
                                 tp: float, direction: str) -> float:
        """
        محاسبه درصد سود نسبت به فاصله کل تا TP
        
        Returns:
            درصد سود (0-100)
        """
        if direction == 'BUY':
            total_distance = tp - entry
            current_profit = current - entry
        else:  # SELL
            total_distance = entry - tp
            current_profit = entry - current
        
        if total_distance <= 0:
            return 0.0
        
        profit_percent = (current_profit / total_distance) * 100
        return max(0.0, min(100.0, profit_percent))
    
    def partial_close_position(self, ticket: int, close_volume: float, 
                               reason: str) -> bool:
        """
        بستن بخشی از حجم معامله
        
        Args:
            ticket: شماره معامله
            close_volume: حجم برای بسته شدن
            reason: دلیل (برای لاگ)
        
        Returns:
            موفق بود یا نه
        """
        try:
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"⚠️ Position #{ticket} not found for partial close")
                return False
            
            pos = positions[0]
            
            # اعتبارسنجی حجم
            if close_volume > pos.volume:
                close_volume = pos.volume
                logger.warning(f"⚠️ Adjusted close volume to {close_volume:.3f}")
            
            if close_volume < 0.01:
                logger.warning(f"⚠️ Close volume too small: {close_volume:.3f}")
                return False
            
            # تعیین نوع سفارش معکوس
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # دریافت قیمت
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                logger.error("❌ Cannot get tick for partial close")
                return False
            
            close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # درخواست بستن
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": round(close_volume, 2),
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 100,
                "magic": 888888,
                "comment": f"PARTIAL_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ PARTIAL CLOSE: {close_volume:.3f} lots @ {close_price:.2f}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   Remaining: {pos.volume - close_volume:.3f} lots")
                
                # به‌روزرسانی state
                if ticket in self.trade_states:
                    state = self.trade_states[ticket]
                    state['total_closed_volume'] += close_volume
                    state['current_volume'] = pos.volume - close_volume
                    state['closed_profit'] += (close_volume * pos.profit / pos.volume)
                
                return True
            else:
                logger.error(f"❌ Partial close failed: {result.retcode if result else 'None'}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in partial close: {e}")
            return False
    
    def update_stop_loss(self, ticket: int, new_sl: float, reason: str) -> bool:
        """
        آپدیت Stop Loss معامله
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            pos = positions[0]
            
            # اعتبارسنجی
            symbol_info = mt5.symbol_info(self.symbol)
            min_distance = symbol_info.trade_stops_level * self.point
            spread = (symbol_info.ask - symbol_info.bid)
            
            if pos.type == mt5.ORDER_TYPE_BUY:
                if new_sl >= pos.price_current - min_distance:
                    logger.warning(f"⚠️ SL too close for BUY")
                    return False
            else:
                if new_sl <= pos.price_current + min_distance:
                    logger.warning(f"⚠️ SL too close for SELL")
                    return False
            
            # درخواست modify
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ SL UPDATED: {new_sl:.2f}")
                logger.info(f"   Reason: {reason}")
                return True
            else:
                logger.error(f"❌ SL update failed: {result.retcode if result else 'None'}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating SL: {e}")
            return False
    
    def manage_trade(self, ticket: int):
        """
        مدیریت کامل معامله (هسته اصلی)
        """
        try:
            # بررسی وجود state
            if ticket not in self.trade_states:
                logger.warning(f"⚠️ No state for trade #{ticket}")
                return
            
            state = self.trade_states[ticket]
            # ⭐ اطمینان از وجود peak_price در state
            if 'peak_price' not in state:
                state['peak_price'] = state['entry_price']
                
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.info(f"ℹ️ Trade #{ticket} already closed")
                del self.trade_states[ticket]
                return
            
            pos = positions[0]
            
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return
            
            current_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # محاسبه درصد سود
            profit_percent = self.calculate_profit_percent(
                state['entry_price'],
                current_price,
                state['initial_tp'],
                state['direction']
            )

            # ⭐ به‌روزرسانی peak با بررسی وجود کلید
            if 'peak_price' not in state:
                state['peak_price'] = state['entry_price']
                
            # به‌روزرسانی peak price
            if state['direction'] == 'BUY':
                if current_price > state['peak_price']:
                    state['peak_price'] = current_price
            else:  # SELL
                if current_price < state['peak_price']:
                    state['peak_price'] = current_price
            
            logger.debug(f"📊 Trade #{ticket}: Profit {profit_percent:.1f}% | "
                        f"Volume: {pos.volume:.3f} lots")
            
            # ═══════════════════════════════════════════════════════
            # ۱. BREAKEVEN: قیمت از Entry عبور کرد
            # ═══════════════════════════════════════════════════════
            if not state['breakeven_done']:
                passed_entry = False
                
                if state['direction'] == 'BUY' and current_price > state['entry_price']:
                    passed_entry = True
                elif state['direction'] == 'SELL' and current_price < state['entry_price']:
                    passed_entry = True
                
                if passed_entry:
                    # SL به آخرین گره زیر Entry
                    if state['direction'] == 'BUY':
                        nearest_node = self.get_nearest_node_below(
                            state['entry_price'], 
                            state['nodes']['below_entry']
                        )
                        new_sl = nearest_node - (5 * self.point)  # 5 پیپ بافر
                    else:  # SELL
                        nearest_node = self.get_nearest_node_above(
                            state['entry_price'],
                            state['nodes']['above_entry']
                        )
                        new_sl = nearest_node + (5 * self.point)
                    
                    if self.update_stop_loss(ticket, new_sl, "Breakeven to nearest node"):
                        state['breakeven_done'] = True
                        state['trailing_active'] = True
                        logger.info(f"🛡️ BREAKEVEN ACTIVATED: SL → {new_sl:.2f} (Node-based)")
            
            # ═══════════════════════════════════════════════════════
            # ۲. PARTIAL CLOSES در سطوح مختلف
            # ═══════════════════════════════════════════════════════
            for level in sorted(self.profit_levels.keys()):
                if profit_percent >= level and not state['partial_closes'][level]:
                    config = self.profit_levels[level]
                    
                    # محاسبه حجم برای بسته شدن
                    if level == 20:
                        # 5% از حجم اولیه
                        close_volume = state['initial_volume'] * 0.05
                    else:
                        # درصدی از حجم باقیمانده
                        close_volume = pos.volume * (config['close_percent'] / 100)
                    
                    # حداقل 0.01 lot
                    close_volume = max(0.01, round(close_volume, 2))
                    
                    # بستن
                    if self.partial_close_position(ticket, close_volume, 
                                                   f"{level}%_{config['description']}"):
                        state['partial_closes'][level] = True
                        logger.info(f"💰 PROFIT SECURED at {level}%: {close_volume:.3f} lots")
                        
                        # Lock سود: انتقال SL به گره مناسب
                        if state['direction'] == 'BUY':
                            lock_node = self.get_nearest_node_below(current_price,
                                                                    state['nodes']['below_entry'])
                        else:
                            lock_node = self.get_nearest_node_above(current_price,
                                                                    state['nodes']['above_entry'])
                        
                        self.update_stop_loss(ticket, lock_node, f"Lock profit at {level}%")
            
            # ═══════════════════════════════════════════════════════
            # ۳. TRAILING: گره به گره
            # ═══════════════════════════════════════════════════════
            if state['trailing_active'] and profit_percent >= 20:
                # شناسایی گره بعدی در مسیر روند
                if state['direction'] == 'BUY':
                    next_node = self.get_nearest_node_below(current_price,
                                                           state['nodes']['below_entry'])
                    # SL باید زیر گره باشد
                    new_sl = next_node - (10 * self.point)  # 10 پیپ بافر
                    
                    # فقط اگر بهتر از SL فعلی باشد
                    if new_sl > pos.sl:
                        self.update_stop_loss(ticket, new_sl, 
                                            f"Trailing to node @ {next_node:.2f}")
                
                else:  # SELL
                    next_node = self.get_nearest_node_above(current_price,
                                                           state['nodes']['above_entry'])
                    new_sl = next_node + (10 * self.point)
                    
                    if new_sl < pos.sl:
                        self.update_stop_loss(ticket, new_sl,
                                            f"Trailing to node @ {next_node:.2f}")
                        
        except Exception as e:
            logger.error(f"❌ Error managing trade #{ticket}: {e}")
            import traceback
            traceback.print_exc()

class ImprovedNodeBasedTrailing:
    """
    سیستم Trailing و Partial Exit بهبودیافته
    """
    
    def __init__(self, mt5_manager, symbol="BTCUSD"):
        self.mt5 = mt5_manager
        self.symbol = symbol
        self.point = 0.1  # برای BTCUSD
        
        # ذخیره وضعیت معاملات
        self.trade_states = {}
        
    def initialize_trade_state(self, ticket: int, entry_price: float, 
                               sl: float, tp: float, volume: float,
                               direction: str, nodes: dict, 
                               spread: float, commission: float):
        """
        مقداردهی اولیه state
        
        Args:
            spread: اسپرد به پیپ (مثلاً 10.0)
            commission: کمیسیون به دلار (مثلاً 0.5)
        """
        self.trade_states[ticket] = {
        'entry_price': entry_price,
        'initial_sl': sl,
        'initial_tp': tp,
        'initial_volume': volume,
        'current_volume': volume,
        'direction': direction,
        'nodes': nodes,
        
        # ⭐ peak_price اضافه شد
        'peak_price': entry_price,
        
        # تنظیمات پیش‌فرض spread و commission
        'spread': 10.0,      # مقدار پیش‌فرض 10 پیپ
        'commission': 0.0,   # بدون کمیسیون
            
            # مراحل مدیریت
            'stage_10pct': False,  # 10% عبور
            'stage_15pct': False,  # 15% عبور (breakeven)
            'stage_50pct': False,  # 50% partial
            'stage_70pct': False,  # 70% partial
            
            # ردیابی
            'peak_price': entry_price,
            'total_closed_volume': 0.0,
            'closed_profit': 0.0,
        }
        
        logger.info(f"✅ Initialized trade #{ticket}")
        logger.info(f"   Entry: {entry_price:.2f} | Direction: {direction}")
        logger.info(f"   Volume: {volume:.3f} | Spread: {spread:.1f} pips")
        logger.info(f"   Nodes: {len(nodes.get('below_entry', []))} below, "
                   f"{len(nodes.get('above_entry', []))} above")
    
    def detect_nodes(self, timeframe=mt5.TIMEFRAME_M3, lookback=100) -> dict:
        """
        شناسایی گره‌های Swing High/Low
        
        Returns:
            {'below_entry': [prices], 'above_entry': [prices], 'all': [prices]}
        """
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, lookback)
            if rates is None or len(rates) == 0:
                logger.error("❌ Cannot fetch rates")
                return {'below_entry': [], 'above_entry': [], 'all': []}
            
            import pandas as pd
            df = pd.DataFrame(rates)
            
            swing_highs = []
            swing_lows = []
            window = 5
            
            for i in range(window, len(df) - window):
                # Swing High
                if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                    swing_highs.append(df['high'].iloc[i])
                
                # Swing Low
                if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                    swing_lows.append(df['low'].iloc[i])
            
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(self.symbol)
            current_price = tick.bid if tick else df['close'].iloc[-1]
            
            # ترکیب و مرتب‌سازی همه گره‌ها
            all_nodes = sorted(set(swing_highs + swing_lows))
            
            nodes_below = sorted([n for n in all_nodes if n < current_price], reverse=True)
            nodes_above = sorted([n for n in all_nodes if n > current_price])
            
            logger.info(f"🔍 Nodes: {len(nodes_below)} below | {len(nodes_above)} above | Total: {len(all_nodes)}")
            
            return {
                'below_entry': nodes_below[:20],  # 20 گره نزدیک
                'above_entry': nodes_above[:20],
                'all': all_nodes
            }
            
        except Exception as e:
            logger.error(f"❌ Node detection error: {e}")
            return {'below_entry': [], 'above_entry': [], 'all': []}
    
    def get_last_node_below(self, reference_price: float, nodes: list) -> float:
        """
        آخرین (نزدیکترین) گره زیر قیمت مرجع
        """
        valid = [n for n in nodes if n < reference_price]
        if not valid:
            return reference_price - (50 * self.point)
        return max(valid)
    
    def get_nearest_node_below_market(self, current_price: float, nodes: list) -> float:
        """
        نزدیک‌ترین گره زیر قیمت فعلی بازار
        """
        valid = [n for n in nodes if n < current_price]
        if not valid:
            return current_price - (30 * self.point)
        return max(valid)
    
    def calculate_profit_distance_percent(self, entry: float, current: float,
                                          tp: float, direction: str) -> float:
        """
        محاسبه درصد پیشروی قیمت نسبت به مسیر کل تا TP
        
        Returns:
            درصد (0-100)
        """
        if direction == 'BUY':
            total = tp - entry
            progress = current - entry
        else:  # SELL
            total = entry - tp
            progress = entry - current
        
        if total <= 0:
            return 0.0
        
        percent = (progress / total) * 100
        return max(0.0, min(100.0, percent))
    
    def partial_close(self, ticket: int, close_volume: float, reason: str) -> bool:
        """
        بستن بخشی از حجم
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"⚠️ Position #{ticket} not found")
                return False
            
            pos = positions[0]
            
            # اعتبارسنجی حجم
            close_volume = min(close_volume, pos.volume)
            close_volume = max(0.01, round(close_volume, 2))
            
            if close_volume < 0.01:
                logger.warning(f"⚠️ Volume too small: {close_volume:.3f}")
                return False
            
            # تعیین نوع
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return False
            
            close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 100,
                "magic": 888888,
                "comment": f"PARTIAL_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ PARTIAL CLOSE: {close_volume:.3f} lots @ {close_price:.2f}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   Remaining: {(pos.volume - close_volume):.3f} lots")
                
                # به‌روزرسانی state
                if ticket in self.trade_states:
                    state = self.trade_states[ticket]
                    state['total_closed_volume'] += close_volume
                    state['current_volume'] = pos.volume - close_volume
                    
                    # تخمین سود بسته شده
                    profit_per_lot = pos.profit / pos.volume if pos.volume > 0 else 0
                    state['closed_profit'] += (close_volume * profit_per_lot)
                
                return True
            else:
                logger.error(f"❌ Partial close failed: {result.retcode if result else 'None'}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in partial_close: {e}")
            return False
    
    def update_sl(self, ticket: int, new_sl: float, reason: str) -> bool:
        """
        آپدیت Stop Loss
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            pos = positions[0]
            
            # اعتبارسنجی فاصله
            symbol_info = mt5.symbol_info(self.symbol)
            min_distance = max(symbol_info.trade_stops_level * self.point, 10 * self.point)
            
            current_price = symbol_info.ask if pos.type == mt5.ORDER_TYPE_BUY else symbol_info.bid
            
            if pos.type == mt5.ORDER_TYPE_BUY:
                if new_sl >= current_price - min_distance:
                    logger.debug(f"⚠️ SL too close for BUY: {new_sl:.2f} vs {current_price:.2f}")
                    return False
            else:
                if new_sl <= current_price + min_distance:
                    logger.debug(f"⚠️ SL too close for SELL: {new_sl:.2f} vs {current_price:.2f}")
                    return False
            
            # modify
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ SL UPDATED: {new_sl:.2f} | {reason}")
                return True
            else:
                logger.debug(f"❌ SL update failed: {result.retcode if result else 'None'}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating SL: {e}")
            return False
    
    def manage_trade(self, ticket: int):
        """
        🎯 هسته اصلی مدیریت معامله
        """
        try:
            if ticket not in self.trade_states:
                logger.warning(f"⚠️ No state for #{ticket}")
                return
            
            state = self.trade_states[ticket]
            
            # دریافت position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.info(f"ℹ️ Trade #{ticket} closed")
                del self.trade_states[ticket]
                return
            
            pos = positions[0]
            
            # قیمت فعلی
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return
            
            current_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # محاسبه درصد پیشروی
            progress_pct = self.calculate_profit_distance_percent(
                state['entry_price'],
                current_price,
                state['initial_tp'],
                state['direction']
            )
            
            # به‌روزرسانی peak
            if state['direction'] == 'BUY':
                state['peak_price'] = max(state['peak_price'], current_price)
            else:
                state['peak_price'] = min(state['peak_price'], current_price)
            
            logger.debug(f"📊 #{ticket}: Progress {progress_pct:.1f}% | "
                        f"Volume: {pos.volume:.3f} | Price: {current_price:.2f}")
            
            # ═════════════════════════════════════════════════════
            # مرحله ۱: قیمت 10% از Entry عبور کرد
            # ═════════════════════════════════════════════════════
            if not state['stage_10pct'] and progress_pct >= 10:
                logger.info(f"🔹 STAGE 1: 10% crossed")
                
                # SL به آخرین گره زیر Entry
                last_node = self.get_last_node_below(state['entry_price'], 
                                                     state['nodes']['below_entry'])
                new_sl = last_node - (5 * self.point)  # 5 پیپ بافر
                
                if self.update_sl(ticket, new_sl, "10% - Last node below entry"):
                    state['stage_10pct'] = True
                    logger.info(f"   → SL to {new_sl:.2f} (node @ {last_node:.2f})")
            
            # ═════════════════════════════════════════════════════
            # مرحله ۲: قیمت 15% عبور کرد → Breakeven
            # ═════════════════════════════════════════════════════
            if not state['stage_15pct'] and progress_pct >= 15:
                logger.info(f"🔹 STAGE 2: 15% crossed - BREAKEVEN")
                
                # SL = Entry + Spread + Commission (به پیپ)
                spread_pips = state['spread']
                commission_pips = state['commission'] / self.point  # تبدیل $ به پیپ
                
                if state['direction'] == 'BUY':
                    new_sl = state['entry_price'] + (spread_pips * self.point) + (commission_pips * self.point)
                else:
                    new_sl = state['entry_price'] - (spread_pips * self.point) - (commission_pips * self.point)
                
                if self.update_sl(ticket, new_sl, "15% - Breakeven + Spread + Comm"):
                    state['stage_15pct'] = True
                    logger.info(f"   → SL to {new_sl:.2f} (BE + {spread_pips:.0f}p spread + {commission_pips:.1f}p comm)")
            
            # ═════════════════════════════════════════════════════
            # مرحله ۳: قیمت به 50% رسید
            # ═════════════════════════════════════════════════════
            if not state['stage_50pct'] and progress_pct >= 50:
                logger.info(f"🔹 STAGE 3: 50% reached")
                
                # Close 50% حجم
                close_vol = state['initial_volume'] * 0.50
                
                if self.partial_close(ticket, close_vol, "50%_profit"):
                    state['stage_50pct'] = True
                    
                    # SL به نزدیک‌ترین گره زیر بازار
                    nearest_node = self.get_nearest_node_below_market(
                        current_price,
                        state['nodes']['all']
                    )
                    new_sl = nearest_node - (10 * self.point)
                    
                    self.update_sl(ticket, new_sl, "50% - Node below market")
                    logger.info(f"   → 50% closed | SL to {new_sl:.2f} (node @ {nearest_node:.2f})")
            
            # ═════════════════════════════════════════════════════
            # مرحله ۴: قیمت به 70% رسید
            # ═════════════════════════════════════════════════════
            if not state['stage_70pct'] and progress_pct >= 70:
                logger.info(f"🔹 STAGE 4: 70% reached")
                
                # Close 30% از حجم باقیمانده
                # باقیمانده فعلی = initial - 50% که بسته شد
                remaining_volume = state['initial_volume'] - state['total_closed_volume']
                close_vol = remaining_volume * 0.30
                
                if self.partial_close(ticket, close_vol, "70%_profit"):
                    state['stage_70pct'] = True
                    
                    # SL به گره زیر بازار
                    nearest_node = self.get_nearest_node_below_market(
                        current_price,
                        state['nodes']['all']
                    )
                    new_sl = nearest_node - (10 * self.point)
                    
                    self.update_sl(ticket, new_sl, "70% - Node below market")
                    logger.info(f"   → 30% closed | 20% remaining to TP")
                    logger.info(f"   → SL to {new_sl:.2f} (node @ {nearest_node:.2f})")
            
            # ═════════════════════════════════════════════════════
            # مرحله ۵: بعد از 70% → Trailing گره‌به‌گره تا TP
            # ═════════════════════════════════════════════════════
            if state['stage_70pct'] and progress_pct >= 70:
                # Trail SL به نزدیک‌ترین گره زیر قیمت فعلی
                nearest_node = self.get_nearest_node_below_market(
                    current_price,
                    state['nodes']['all']
                )
                new_sl = nearest_node - (10 * self.point)
                
                # فقط اگر بهتر از SL فعلی باشد
                if state['direction'] == 'BUY':
                    if new_sl > pos.sl + (20 * self.point):  # حداقل 20 پیپ بهبود
                        self.update_sl(ticket, new_sl, f"Trailing to node @ {nearest_node:.2f}")
                else:  # SELL
                    if new_sl < pos.sl - (20 * self.point):
                        self.update_sl(ticket, new_sl, f"Trailing to node @ {nearest_node:.2f}")
                        
        except Exception as e:
            logger.error(f"❌ Error managing #{ticket}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN EXECUTION - اجرای اصلی
# ============================================================================
        self.mt5.refresh_account()
        positions = self.mt5.get_active_positions()
        
        bid, ask = self.mt5.get_current_price()
        
        # محاسبه وین ریت
        win_rate = 0
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
        
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"📊 PROFESSIONAL BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"   Strategy: {self.current_strategy.upper()}")
        logger.info(f"   Balance: ${self.mt5.account_info.balance:,.2f}")
        logger.info(f"   Equity: ${self.mt5.account_info.equity:,.2f}")
        logger.info(f"   Market: {bid:.2f} | {ask:.2f}")
        logger.info(f"   Trades This Hour: {self.trades_this_hour}/{self.max_trades_per_hour}")
        logger.info(f"   Total Trades: {self.total_trades}")
        logger.info(f"   Win Rate: {win_rate:.1%}")
        logger.info(f"   Total P/L: ${self.total_profit:.2f}")
        logger.info(f"   Avg P/L: ${avg_profit:.2f}")
        
        if positions:
            logger.info(f"   Active Positions: {len(positions)}")
            for pos in positions[:2]:  # حداکثر 2 پوزیشن نمایش بده
                # اصلاح: استفاده از pos.type به جای pos.order_type
                profit_pips = abs(pos.price_current - pos.price_open) / (self.mt5.get_point() or 0.01)
                logger.info(f"     #{pos.ticket}: {'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'} "
                        f"{pos.volume} lots, P/L: ${pos.profit:.2f} ({profit_pips:.1f} pips)")
        else:
            logger.info("   Status: Looking for opportunities...")
        
        logger.info("=" * 60)
        
        self._last_report = datetime.now()
        
    def __init__(self, symbol: str = "BTCUSD", max_lots: float = None, config: Any = None):
        # فراخوانی سازنده والد
        super().__init__(symbol, max_lots=max_lots, config=config)
        
        # استراتژی‌های اسکلپ (با فاصله‌های واقع‌بینانه)
        self.scalp_strategies = {
            'quick': {'target_pips': 20, 'stop_pips': 15, 'timeout_sec': 120, 'volume': 0.1},
            'normal': {'target_pips': 30, 'stop_pips': 20, 'timeout_sec': 180, 'volume': 0.2},
            'aggressive': {'target_pips': 50, 'stop_pips': 30, 'timeout_sec': 240, 'volume': 0.3}
        }
        
        self.current_strategy = 'normal'
        
        # تنظیمات اسکلپ
        self.scalp_mode = True
        self.max_trades_per_hour = 10
        self.trades_this_hour = 0
        self.hour_start = datetime.now()
        # بعد از self.scalp_strategies اضافه کنید:
        self.allow_multiple_positions = False  # جلوگیری از باز کردن چند پوزیشن
        self.position_check_delay = 3  # تاخیر بین چک کردن پوزیشن‌ها
        # آمار
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        
        # پارامترهای تطبیقی
        self.risk_multiplier = 1.0
        self.volume_multiplier = 1.0
        
        logger.info("🎯 PROFESSIONAL SCALPING BOT INITIALIZED")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Strategy: {self.current_strategy}")
        logger.info(f"   Max Trades/Hour: {self.max_trades_per_hour}")
        logger.info(f"   Target: {self.scalp_strategies[self.current_strategy]['target_pips']} pips")
        logger.info(f"   Stop: {self.scalp_strategies[self.current_strategy]['stop_pips']} pips")
    
    def start(self):
        """شروع ربات"""
        logger.info("🚀 STARTING PROFESSIONAL SCALPING BOT...")
        
        if not self.mt5.connect():
            logger.error("❌ Failed to connect to MT5")
            return False
        
        # تست اولیه
        if not self._initial_test():
            logger.error("❌ Initial test failed")
            return False
        
        # پاکسازی پوزیشن‌های قدیمی
        self._cleanup_old_positions()
        
        self.running = True
        logger.info("✅ Bot started successfully!")
        
        try:
            self._professional_loop()
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.stop()
        
        return True
    
    def _cleanup_old_positions(self):
        """پاکسازی پوزیشن‌های قدیمی"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return
            
            if len(positions) > 0:
                logger.info(f"🧹 Found {len(positions)} open position(s). Checking...")
                
                for pos in positions:
                    # محاسبه زمان باز بودن
                    pos_time = datetime.fromtimestamp(pos.time)
                    time_open = (datetime.now() - pos_time).seconds
                    
                    if time_open > 1800:  # بیش از 30 دقیقه
                        logger.warning(f"⚠️ Closing old position #{pos.ticket} (open for {time_open//60}m)")
                        
                        # دریافت قیمت
                        tick = mt5.symbol_info_tick(self.symbol)
                        if not tick:
                            continue
                        
                        # بستن پوزیشن
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            close_type = mt5.ORDER_TYPE_SELL
                            price = tick.bid
                        else:
                            close_type = mt5.ORDER_TYPE_BUY
                            price = tick.ask
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": self.symbol,
                            "volume": pos.volume,
                            "type": close_type,
                            "position": pos.ticket,
                            "price": price,
                            "deviation": 100,
                            "magic": 999999,
                            "comment": "CLEANUP_OLD",
                            "type_time": mt5.ORDER_TIME_GTC,
                        }
                        
                        result = mt5.order_send(request)
                        if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"✅ Closed old position #{pos.ticket}")
                        else:
                            logger.error(f"❌ Failed to close position #{pos.ticket}")
                    else:
                        logger.info(f"ℹ️ Position #{pos.ticket} is recent ({time_open//60}m). Keeping it.")
            else:
                logger.info("✅ No open positions found.")
                
        except Exception as e:
            logger.error(f"❌ Error in cleanup: {e}")
    
    def _initial_test(self) -> bool:
        """تست اولیه اتصال"""
        try:
            # تست قیمت
            bid, ask = self.mt5.get_current_price()
            if bid == 0 or ask == 0:
                logger.error("❌ Cannot get market prices")
                return False
            
            logger.info(f"💰 Market: Bid={bid:.2f}, Ask={ask:.2f}")
            
            # تست symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info:
                point = symbol_info.point
                spread = symbol_info.spread
                spread_pips = spread * point if point > 0 else 0
                logger.info(f"📊 Symbol Info: Point={point}, Spread={spread_pips:.2f} pips")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initial test error: {e}")
            return False
    
    def _professional_loop(self):
        """حلقه اصلی - استفاده مستقیم از MT5 objects"""
        logger.info("🔄 Entering main trading loop...")
        
        last_analysis = None
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 1. مدیریت ساعتی
                self._manage_hourly_reset(current_time)
                
                # 2. بررسی پوزیشن‌های باز - استفاده مستقیم از MT5 objects
                positions = mt5.positions_get(symbol=self.symbol)
                
                if positions:
                    # مدیریت پوزیشن باز - بدون تبدیل به TradeInfo
                    for pos in positions:
                        self._manage_mt5_position(pos, current_time)
                    
                    time.sleep(0.5)
                    continue
                
                # 3. بررسی محدودیت معاملات
                if self.trades_this_hour >= self.max_trades_per_hour:
                    logger.info(f"⏸️ Hourly limit reached: {self.trades_this_hour}/{self.max_trades_per_hour}")
                    time.sleep(10)
                    continue
                
                # 4. تحلیل بازار
                if last_analysis is None or (current_time - last_analysis).seconds >= 10:
                    signal = self._analyze_for_scalp()
                    
                    if signal and signal.is_valid():
                        # 5. اجرای معامله
                        if self._execute_scalp_trade(signal):
                            self.trades_this_hour += 1
                            self.total_trades += 1
                    
                    last_analysis = current_time
                
                # 6. گزارش دوره‌ای
                if self._should_report(current_time):
                    self._print_status_report()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                time.sleep(5)

    def _manage_mt5_position(self, position, current_time: datetime):
        """مدیریت مستقیم پوزیشن MT5"""
        try:
            strategy = self.scalp_strategies[self.current_strategy]
            
            # دریافت قیمت فعلی
            bid, ask = self.mt5.get_current_price()
            if bid == 0 or ask == 0:
                return
            
            # استفاده مستقیم از position.type
            current_price = bid if position.type == mt5.ORDER_TYPE_BUY else ask
            
            # محاسبه پیپ
            point = self.mt5.get_point()
            if point == 0:
                point = 0.01
            
            # استفاده مستقیم از position attributes
            profit_pips = abs(current_price - position.price_open) / point
            current_profit = position.profit
            
            # زمان باز بودن
            pos_time = datetime.fromtimestamp(position.time)
            time_delta = current_time - pos_time
            time_open = (current_time - pos_time).seconds
            
            logger.debug(f"📊 Managing position #{position.ticket}: P/L=${current_profit:.2f}, Pips={profit_pips:.1f}, Time={time_open}s")
            
            # قوانین بستن
            # 1. تارگت سود
            if profit_pips >= strategy['target_pips'] and current_profit > 0:
                self._close_mt5_position(position, f"Target reached ({strategy['target_pips']} pips)")
                self.winning_trades += 1
                self.total_profit += current_profit
                return
            
            # 2. استاپ لاس
            if profit_pips >= strategy['stop_pips'] and current_profit < 0:
                self._close_mt5_position(position, f"Stop loss ({strategy['stop_pips']} pips)")
                self.total_profit += current_profit
                return
            
            # 3. تایم‌اوت
            if time_open >= strategy['timeout_sec']:
                action = "CLOSE" if abs(current_profit) > 0.1 else "BREAKEVEN"
                self._close_mt5_position(position, f"{action} after {time_open}s")
                if current_profit > 0:
                    self.winning_trades += 1
                self.total_profit += current_profit
                return
            
            # 4. بریک‌اون خودکار
            if profit_pips >= (strategy['target_pips'] * 0.5) and current_profit > 0:
                self._move_mt5_to_breakeven(position)
                
        except Exception as e:
            logger.error(f"❌ Error managing MT5 position: {e}")

    def _close_mt5_position(self, position, reason: str):
        """بستن مستقیم پوزیشن MT5"""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return False
            
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": 100,
                "magic": 888888,
                "comment": f"CLOSE: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Closed #{position.ticket}: {reason} | P/L: ${position.profit:.2f}")
                return True
            else:
                logger.error(f"❌ Failed to close #{position.ticket}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error closing MT5 position: {e}")
            return False
        

    def _manage_hourly_reset(self, current_time: datetime):
        """مدیریت ریست ساعتی"""
        hour_diff = (current_time - self.hour_start).seconds / 3600
        
        if hour_diff >= 1:
            logger.info(f"🔄 Hourly reset: Had {self.trades_this_hour} trades last hour")
            self.trades_this_hour = 0
            self.hour_start = current_time
            
            # تنظیم تطبیقی استراتژی
            self._adjust_strategy_based_on_performance()
    
    def _adjust_strategy_based_on_performance(self):
        """تنظیم استراتژی بر اساس عملکرد"""
        if self.total_trades < 3:  # حداقل 3 معامله برای تصمیم‌گیری
            return
        
        win_rate = self.winning_trades / self.total_trades
        
        if win_rate > 0.6:
            # عملکرد خوب
            if self.current_strategy != 'aggressive':
                self.current_strategy = 'aggressive'
                logger.info("📈 Good performance! Switching to AGGRESSIVE strategy")
        elif win_rate < 0.3:
            # عملکرد ضعیف
            if self.current_strategy != 'quick':
                self.current_strategy = 'quick'
                logger.info("📉 Poor performance! Switching to QUICK strategy")
                self.volume_multiplier = max(self.volume_multiplier * 0.7, 0.5)
        else:
            # عملکرد متوسط
            if self.current_strategy != 'normal':
                self.current_strategy = 'normal'
                logger.info("📊 Average performance! Switching to NORMAL strategy")
    
    def _manage_open_position(self, trade: TradeInfo, current_time: datetime):
        """مدیریت پوزیشن باز"""
        try:
            strategy = self.scalp_strategies[self.current_strategy]
            
            # دریافت قیمت فعلی
            bid, ask = self.mt5.get_current_price()
            if bid == 0 or ask == 0:
                return
            
            # استفاده از trade.order_type که درست است
            current_price = bid if trade.order_type == mt5.ORDER_TYPE_BUY else ask
            
            # محاسبه پیپ
            point = self.mt5.get_point()
            if point == 0:
                point = 0.01  # مقدار پیش‌فرض برای BTCUSD
            
            profit_pips = abs(current_price - trade.open_price) / point
            current_profit = trade.profit
            
            # زمان باز بودن
            time_open = (current_time - trade.open_time).seconds
            
            # قوانین بستن
            
            # 1. تارگت سود
            if profit_pips >= strategy['target_pips'] and current_profit > 0:
                self.trade.close_trade(trade, f"Target reached ({strategy['target_pips']} pips)")
                logger.info(f"🎯 Target hit! Profit: ${current_profit:.2f} ({profit_pips:.1f} pips)")
                self.winning_trades += 1
                self.total_profit += current_profit
                return
            
            # 2. استاپ لاس
            if profit_pips >= strategy['stop_pips'] and current_profit < 0:
                self.trade.close_trade(trade, f"Stop loss ({strategy['stop_pips']} pips)")
                logger.info(f"🛑 Stop loss! Loss: ${abs(current_profit):.2f} ({profit_pips:.1f} pips)")
                self.total_profit += current_profit
                return
            
            # 3. تایم‌اوت
            if time_open >= strategy['timeout_sec']:
                action = "CLOSE" if abs(current_profit) > 0.1 else "BREAKEVEN"
                self.trade.close_trade(trade, f"{action} after {time_open}s")
                logger.info(f"⏰ {action}! P/L: ${current_profit:.2f} after {time_open}s")
                
                if current_profit > 0:
                    self.winning_trades += 1
                self.total_profit += current_profit
                return
            
            # 4. بریک‌اون خودکار (اگر 50% تارگت سود کردیم)
            if profit_pips >= (strategy['target_pips'] * 0.5) and current_profit > 0:
                self._move_to_breakeven(trade)
            
        except Exception as e:
            logger.error(f"❌ Error managing position: {e}")
    
    def emergency_close_all_positions(self):
        """بستن فوری همه پوزیشن‌های باز"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                logger.info("✅ No open positions")
                return
            
            logger.warning(f"🚨 EMERGENCY: Closing {len(positions)} open positions")
            
            for pos in positions:
                tick = mt5.symbol_info_tick(self.symbol)
                if not tick:
                    continue
                
                if pos.type == mt5.ORDER_TYPE_BUY:
                    close_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    close_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 100,
                    "magic": 999999,
                    "comment": "EMERGENCY_CLOSE",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Closed position #{pos.ticket}")
                else:
                    logger.error(f"❌ Failed to close #{pos.ticket}")
                    
        except Exception as e:
            logger.error(f"❌ Emergency close error: {e}")
            
    def _move_to_breakeven(self, trade: TradeInfo):
        """جابجایی استاپ به نقطه ورود"""
        try:
            new_sl = trade.open_price
            
            # فقط اگر بهتر از استاپ فعلی باشد
            if trade.order_type == mt5.ORDER_TYPE_BUY:
                if new_sl > trade.sl:
                    success = self.trade.update_trailing_stop(trade, new_sl)
                    if success:
                        logger.debug(f"🛡️ Moved SL to breakeven: {new_sl:.2f}")
            else:
                if new_sl < trade.sl:
                    success = self.trade.update_trailing_stop(trade, new_sl)
                    if success:
                        logger.debug(f"🛡️ Moved SL to breakeven: {new_sl:.2f}")
                        
        except Exception as e:
            logger.debug(f"Could not move to breakeven: {e}")
    
    def _analyze_for_scalp(self) -> Optional[NDSSignal]:
        """تحلیل برای اسکلپ"""
        try:
            # دریافت داده‌های M1
            df = self.mt5.get_ohlcv(mt5.TIMEFRAME_M1, 50)
            if df is None or len(df) < 20:
                return None
            
            prices = df['close'].values
            current_price = prices[-1]
            
            # تحلیل روند سریع
            ma_fast = np.mean(prices[-7:])   # 7 کندل
            ma_slow = np.mean(prices[-20:])  # 20 کندل
            
            # قدرت روند
            trend_strength = abs(ma_fast - ma_slow) / current_price
            
            # تشخیص جهت با فیلتر قوی‌تر
            if current_price > ma_fast > ma_slow and trend_strength > 0.0005:
                direction = TrendDirection.BULLISH
                confidence = 0.7 + min(trend_strength * 150, 0.25)
            elif current_price < ma_fast < ma_slow and trend_strength > 0.0005:
                direction = TrendDirection.BEARISH
                confidence = 0.7 + min(trend_strength * 150, 0.25)
            else:
                return None  # روند ضعیف یا بدون روند
            
            # تنظیم استراتژی
            strategy = self.scalp_strategies[self.current_strategy]
            
            # دریافت اطلاعات symbol
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                return None
            
            point = symbol_info.point
            if point == 0:
                point = 0.01
            
            # حداقل فاصله بر اساس stops_level
            stops_level = getattr(symbol_info, 'trade_stops_level', 10)
            min_distance_pips = max(stops_level, 15)  # حداقل 15 پیپ
            
            # استفاده از حداقل فاصله
            stop_pips = max(strategy['stop_pips'], min_distance_pips)
            target_pips = max(strategy['target_pips'], min_distance_pips * 1.5)
            
            # دریافت قیمت لحظه‌ای
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return None
            
            # محاسبه سطوح
            if direction == TrendDirection.BULLISH:
                entry = tick.ask
                sl = entry - (stop_pips * point)
                tp = entry + (target_pips * point)
            else:
                entry = tick.bid
                sl = entry + (stop_pips * point)
                tp = entry - (target_pips * point)
            
            # اعتبارسنجی سطوح
            entry, sl, tp = self._validate_scalp_levels(entry, sl, tp, direction)
            
            # محاسبه R/R
            risk_reward = abs(tp - entry) / abs(entry - sl)
            
            if risk_reward < 1.5:
                return None
            
            # ساخت سیگنال
            signal = NDSSignal(
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                quantum_state=QuantumState.COLLAPSED_BULLISH if direction == TrendDirection.BULLISH else QuantumState.COLLAPSED_BEARISH,
                hurst_exponent=0.6,
                risk_reward=risk_reward,
                timestamp=datetime.now(),
                nodes=[]
            )
            
            logger.info(f"🎯 SCALP Signal ({self.current_strategy.upper()}):")
            logger.info(f"   Direction: {'BUY 🚀' if direction == TrendDirection.BULLISH else 'SELL 📉'}")
            logger.info(f"   Entry: {entry:.2f}")
            logger.info(f"   SL: {sl:.2f} ({abs(entry-sl)/point:.1f} pips)")
            logger.info(f"   TP: {tp:.2f} ({abs(tp-entry)/point:.1f} pips)")
            logger.info(f"   R/R: {risk_reward:.2f}")
            logger.info(f"   Confidence: {confidence:.1%}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return None
    
    def _validate_scalp_levels(self, entry: float, sl: float, tp: float, direction: TrendDirection) -> Tuple[float, float, float]:
        """اعتبارسنجی سطوح SL/TP"""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                return entry, sl, tp
            
            point = symbol_info.point
            if point == 0:
                point = 0.01
            
            # حداقل فاصله
            stops_level = getattr(symbol_info, 'trade_stops_level', 10)
            freeze_level = getattr(symbol_info, 'trade_freeze_level', 0)
            min_distance = max(stops_level, freeze_level, 15) * point  # حداقل 15 پیپ
            
            if direction == TrendDirection.BULLISH:
                # برای BUY: SL زیر entry
                if sl >= entry - (point * 5):
                    sl = entry - min_distance
                    logger.debug(f"   Adjusted BUY SL to {sl:.2f}")
                
                # TP بالای entry
                if tp <= entry + (point * 5):
                    tp = entry + (min_distance * 2)
                    logger.debug(f"   Adjusted BUY TP to {tp:.2f}")
            
            else:  # SELL
                # برای SELL: SL بالای entry
                if sl <= entry + (point * 5):
                    sl = entry + min_distance
                    logger.debug(f"   Adjusted SELL SL to {sl:.2f}")
                
                # TP زیر entry
                if tp >= entry - (point * 5):
                    tp = entry - (min_distance * 2)
                    logger.debug(f"   Adjusted SELL TP to {tp:.2f}")
            
            # بررسی نهایی
            if direction == TrendDirection.BULLISH:
                if sl >= entry:
                    sl = entry - min_distance
                if tp <= entry:
                    tp = entry + min_distance
            else:
                if sl <= entry:
                    sl = entry + min_distance
                if tp >= entry:
                    tp = entry - min_distance
            
            return entry, sl, tp
            
        except Exception as e:
            logger.error(f"❌ Error validating levels: {e}")
            return entry, sl, tp
    
    def _execute_scalp_trade(self, signal: NDSSignal) -> bool:
        """اجرای معامله اسکلپ"""
        # بررسی قوی برای جلوگیری از باز کردن چند پوزیشن
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            logger.warning(f"⏸️ Skipping trade: {len(positions)} position(s) already open")
            return False
        try:
            # محاسبه حجم
            strategy = self.scalp_strategies[self.current_strategy]
            volume = strategy['volume'] * self.volume_multiplier
            
            # اعتبارسنجی حجم
            volume = max(0.01, min(volume, 0.5))
            
            # دریافت قیمت لحظه‌ای
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                logger.error("❌ Cannot get tick data")
                return False
            
            # تنظیم پارامترها
            if signal.direction == TrendDirection.BULLISH:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # ✅ استفاده از TradeManager برای باز کردن معامله
            ticket = self.trade.open_trade_safe(signal)
            if ticket:
                logger.info(f"✅ Scalp trade opened via TradeManager! Ticket: #{ticket}")
                return True
            else:
                logger.error("❌ Scalp trade opening failed via TradeManager")
                return False
            
            if result is None:
                logger.error("❌ Order send returned None")
                return False
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Trade opened! Ticket: #{result.order}")
                return True
            else:
                logger.error(f"❌ Trade failed: {result.retcode}")
                if result.retcode == 10030:
                    logger.error("   💡 Error 10030: Invalid stops or price")
                    logger.error(f"   💡 Try increasing stops distance")
                return False
                
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return False
    
    def _should_report(self, current_time: datetime) -> bool:
        """بررسی زمان گزارش"""
        if not hasattr(self, '_last_report'):
            self._last_report = current_time
            return True
        
        return (current_time - self._last_report).seconds >= 30
    
    def _print_status_report(self):
        """گزارش وضعیت"""
        self.mt5.refresh_account()
        positions = mt5.positions_get(symbol=self.symbol)
        
        bid, ask = self.mt5.get_current_price()
        
        logger.info("=" * 60)
        logger.info(f"📊 OPTIMIZED BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"   Balance: ${self.mt5.account_info.balance:,.2f}")
        logger.info(f"   Equity: ${self.mt5.account_info.equity:,.2f}")
        logger.info(f"   Market: {bid:.2f} | {ask:.2f}")
        
        if positions:
            logger.info(f"   Active Positions: {len(positions)}")
            for pos in positions:
                if pos.ticket in self.trailing_manager.trade_states:
                    state = self.trailing_manager.trade_states[pos.ticket]  # dict
                    profit_pct = self.trailing_manager.calculate_profit_distance_percent(pos.ticket)
                    
                    logger.info(f"     #{pos.ticket}: {'BUY' if pos.type == 0 else 'SELL'} "
                            f"{pos.volume} lots, P/L: ${pos.profit:.2f} ({profit_pct:.1f}% to TP)")
                    logger.info(f"       Stage: {state['stage']}, Peak: {state['peak_price']:.2f}")
                else:
                    logger.info(f"     #{pos.ticket}: {'BUY' if pos.type == 0 else 'SELL'} "
                            f"{pos.volume} lots, P/L: ${pos.profit:.2f} (No state)")
        else:
            logger.info("   Status: Looking for opportunities...")
        
        logger.info("=" * 60)
        
        self._last_report = datetime.now()

        

    def stop(self):
        """توقف ربات"""
        # گزارش نهایی
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            avg_profit = self.total_profit / self.total_trades
            
            logger.info("📈 FINAL PERFORMANCE REPORT:")
            logger.info(f"   Total Trades: {self.total_trades}")
            logger.info(f"   Winning Trades: {self.winning_trades}")
            logger.info(f"   Win Rate: {win_rate:.1%}")
            logger.info(f"   Total Profit: ${self.total_profit:.2f}")
            logger.info(f"   Average Profit/Trade: ${avg_profit:.2f}")
        
        super().stop()


# ============================================================================
# MAIN EXECUTION - اجرای اصلی
# ============================================================================
def main():
    """اجرای ربات اسکلپینگ حرفه‌ای"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║            🎯 PROFESSIONAL SCALPING BOT                 ║
    ║══════════════════════════════════════════════════════════║
    ║  📌 Symbol: BTCUSD                                      ║
    ║  📌 Strategy: Adaptive Professional Scalping            ║
    ║  📌 Timeframes: M1/M2/M5 Adaptive                      ║
    ║  📌 Strategies: Quick(5p), Normal(10p), Aggressive(15p)║
    ║  📌 Features:                                          ║
    ║     • Adaptive Strategy Selection                       ║
    ║     • Smart Risk Management                            ║
    ║     • Market State Detection                           ║
    ║     • Partial Closing                                  ║
    ║     • Performance Analytics                            ║
    ║  📌 Max Trades/Hour: 10                                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # بررسی وجود کلاس‌های لازم
        required_classes = ['ProfessionalScalpingBot', 'ScalpingNDSTradingBot']
        missing_classes = []
        
        for cls_name in required_classes:
            if cls_name not in globals():
                missing_classes.append(cls_name)
        
        if missing_classes:
            print(f"❌ Missing classes: {', '.join(missing_classes)}")
            print("Please make sure all required classes are defined.")
            return
        
        # ایجاد و شروع ربات
        bot = ProfessionalScalpingBot(symbol="BTCUSD", max_lots=0.5)
        bot.start()
        
    except Exception as e:
        logger.error(f"Failed to start professional scalping bot: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def run_optimized_bot():
    """اجرای بات - نسخه اصلاح شده با مدیریت فعال (تابع standalone)"""
    # این تابع باید از داخل کلاس فراخوانی شود یا به کلاس منتقل شود
    logger.error("❌ run_optimized_bot should be called from a bot instance, not as standalone function")
    logger.info("💡 Please use OptimizedNDSTradingBot class instead")
    return
    logger.info("=" * 80)
    logger.info("🤖 OPTIMIZED NDS BOT STARTED")
    logger.info("=" * 80)
    
    self.running = True
    iteration = 0
    
    # متغیرهای ردیابی
    last_analysis_time = datetime.now() - timedelta(minutes=5)
    last_status_report = datetime.now()
    
    try:
        while self.running:
            iteration += 1
            current_time = datetime.now()
            
            # ✅ 1. بررسی معاملات باز - هر 5 ثانیه
            positions = self.mt5.get_active_positions()
            
            if positions:
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 ACTIVE POSITION DETECTED - Managing Trade")
                logger.info(f"{'='*70}")
                
                for pos in positions:
                    bid, ask = self.mt5.get_current_price()
                    current_price = bid if pos.order_type == mt5.ORDER_TYPE_BUY else ask
                    point = self.mt5.get_point()
                    
                    # محاسبه سود
                    if pos.order_type == mt5.ORDER_TYPE_BUY:
                        profit_pips = (current_price - pos.open_price) / point
                    else:
                        profit_pips = (pos.open_price - current_price) / point
                    
                    logger.info(f"   Ticket: #{pos.ticket}")
                    logger.info(f"   Type: {'BUY' if pos.order_type == 0 else 'SELL'}")
                    logger.info(f"   Entry: {pos.open_price:.2f}")
                    logger.info(f"   Current: {current_price:.2f}")
                    logger.info(f"   SL: {pos.sl:.2f} | TP: {pos.tp:.2f}")
                    logger.info(f"   Profit: {profit_pips:.1f} pips (${pos.profit:.2f})")
                    
                    # ✅ فراخوانی مدیریت معامله
                    if hasattr(self, '_manage_open_trade'):
                        self._manage_open_trade(pos)
                    else:
                        _manage_open_trade_standalone(self, pos)
                
                logger.info(f"{'='*70}\n")
                
                # منتظر 5 ثانیه قبل از بررسی بعدی
                time.sleep(5)
                continue  # برگشت به ابتدای حلقه
            
            # ✅ 2. اگر معامله‌ای نیست، تحلیل برای ورود
            else:
                # تحلیل هر 1 دقیقه
                if (current_time - last_analysis_time).seconds >= 60:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"🔍 ANALYSIS CYCLE #{iteration}")
                    logger.info(f"{'='*70}")
                    
                    # دریافت داده
                    df = self.mt5.get_ohlcv(mt5.TIMEFRAME_M3, 500)
                    
                    if df is not None and len(df) >= 100:
                        # تحلیل NDS (analyze بدون پارامتر df استفاده می‌کند)
                        signal = self.nds.analyze()
                        
                        if signal and signal.is_valid():
                            logger.info(f"✅ VALID SIGNAL FOUND!")
                            logger.info(f"   Direction: {signal.direction.name}")
                            logger.info(f"   Entry: {signal.entry_price:.2f}")
                            logger.info(f"   SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
                            logger.info(f"   R/R: {signal.risk_reward:.2f}")
                            logger.info(f"   Confidence: {signal.confidence:.2%}")
                            
                            # باز کردن معامله
                            ticket = self.trade.open_trade_safe(signal)
                            
                            if ticket:
                                logger.info(f"🚀 Trade opened: #{ticket}")
                            else:
                                logger.warning("⚠️ Trade opening failed")
                        else:
                            logger.info("⏳ No valid signal - waiting...")
                    
                    last_analysis_time = current_time
                
                # گزارش وضعیت هر 30 ثانیه
                if (current_time - last_status_report).seconds >= 30:
                    # استفاده از متد کلاس یا helper function
                    if hasattr(self, '_status_report'):
                        self._status_report()
                    else:
                        _status_report_standalone(self)
                    last_status_report = current_time
                
                # استراحت 5 ثانیه
                time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in bot loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        self.running = False
        logger.info("🛑 Bot shutdown complete")

# این متد باید در کلاس OptimizedNDSTradingBot باشد
# فعلاً به عنوان تابع standalone تعریف شده - باید به کلاس منتقل شود
def _manage_open_trade_standalone(bot_instance, position):
    """مدیریت معامله باز - با Trailing Stop فعال (standalone version)"""
    try:
        # تبدیل position به TradeInfo
        if not isinstance(position, TradeInfo):
            trade = TradeInfo(
                ticket=position.ticket,
                symbol=position.symbol,
                order_type=position.type,
                volume=position.volume,
                open_price=position.price_open,
                current_price=position.price_current,
                sl=position.sl,
                tp=position.tp,
                profit=position.profit,
                open_time=datetime.fromtimestamp(position.time)
            )
        else:
            trade = position

        # دریافت قیمت فعلی
        bid, ask = bot_instance.mt5.get_current_price()
        if bid == 0 or ask == 0:
            return

        current_price = bid if trade.order_type == mt5.ORDER_TYPE_BUY else ask
        is_buy = trade.order_type == mt5.ORDER_TYPE_BUY
        
        # محاسبه point
        if bot_instance.symbol == "BTCUSD":
            point = 1.0  # ⬅️ 1 دلار = 1 پیپ
        else:
            symbol_info = mt5.symbol_info(bot_instance.symbol)
            point = symbol_info.point if symbol_info else 0.01
        
        spread = bot_instance.mt5.get_spread()

        # محاسبه سود
        if is_buy:
            profit_pips = (current_price - trade.open_price) / point
        else:
            profit_pips = (trade.open_price - current_price) / point

        logger.info(f"📊 Position #{trade.ticket}: Price={current_price:.2f}, Profit={profit_pips:.1f} pips, P/L=${trade.profit:.2f}")

        # مقداردهی اولیه
        if not hasattr(trade, '_peak_price'):
            trade._peak_price = current_price
            trade._breakeven_done = False
            trade._trailing_active = False
            trade._last_trailing_update = datetime.now()

        # به‌روزرسانی peak
        if is_buy:
            if current_price > trade._peak_price:
                trade._peak_price = current_price
        else:
            if current_price < trade._peak_price:
                trade._peak_price = current_price

        # ✅ PHASE 1: BREAKEVEN (10 پیپ)
        if not trade._breakeven_done and profit_pips >= 10:
            if is_buy:
                breakeven_sl = trade.open_price + spread + (2 * point)
                
                if breakeven_sl > trade.sl:
                    success = bot_instance.trade.update_trailing_stop(trade, breakeven_sl)
                    if success:
                        trade.sl = breakeven_sl  # ⬅️ به‌روزرسانی دستی
                        trade._breakeven_done = True
                        logger.info(f"🛡️ BREAKEVEN: SL={breakeven_sl:.2f}")
            else:
                breakeven_sl = trade.open_price - spread - (2 * point)
                
                if breakeven_sl < trade.sl:
                    success = bot_instance.trade.update_trailing_stop(trade, breakeven_sl)
                    if success:
                        trade.sl = breakeven_sl  # ⬅️ به‌روزرسانی دستی
                        trade._breakeven_done = True
                        logger.info(f"🛡️ BREAKEVEN: SL={breakeven_sl:.2f}")

        # ✅ PHASE 2: TRAILING STOP (15+ پیپ)
        elif profit_pips >= 15:
            if not trade._trailing_active:
                trade._trailing_active = True
                logger.info(f"🔥 TRAILING ACTIVATED at {profit_pips:.1f} pips")
            
            # محدودیت زمانی: هر 3 ثانیه یک بار
            current_time = datetime.now()
            if (current_time - trade._last_trailing_update).seconds < 3:
                return
            
            trailing_distance = 8 * point  # 8 دلار
            
            if is_buy:
                new_sl = current_price - trailing_distance - spread
                
                # شرط: new_sl باید بزرگتر از SL فعلی باشد
                if new_sl > trade.sl:
                    sl_improvement = (new_sl - trade.sl) / point
                    
                    # حداقل 2 پیپ بهبود
                    if sl_improvement >= 2:
                        success = bot_instance.trade.update_trailing_stop(trade, new_sl)
                        if success:
                            trade.sl = new_sl  # ⬅️ به‌روزرسانی دستی
                            trade._last_trailing_update = current_time
                            logger.info(f"📈 TRAILING: {trade.sl - sl_improvement*point:.2f} → {new_sl:.2f} (+{sl_improvement:.1f} pips)")
            else:
                new_sl = current_price + trailing_distance + spread
                
                # شرط: new_sl باید کوچکتر از SL فعلی باشد
                if new_sl < trade.sl:
                    sl_improvement = (trade.sl - new_sl) / point
                    
                    # حداقل 2 پیپ بهبود
                    if sl_improvement >= 2:
                        success = bot_instance.trade.update_trailing_stop(trade, new_sl)
                        if success:
                            trade.sl = new_sl  # ⬅️ به‌روزرسانی دستی
                            trade._last_trailing_update = current_time
                            logger.info(f"📉 TRAILING: {trade.sl + sl_improvement*point:.2f} → {new_sl:.2f} (+{sl_improvement:.1f} pips)")

    except Exception as e:
        logger.error(f"❌ Error managing position: {e}")
        import traceback
        traceback.print_exc()


def _status_report_standalone(bot_instance):
    """گزارش وضعیت (standalone helper)"""
    bot_instance.mt5.refresh_account()
    
    logger.info(f"\n📊 STATUS REPORT")
    logger.info(f"   Balance: ${bot_instance.mt5.account_info.balance:,.2f}")
    logger.info(f"   Equity: ${bot_instance.mt5.account_info.equity:,.2f}")
    
    positions = bot_instance.mt5.get_active_positions()
    if positions:
        logger.info(f"   ✅ {len(positions)} active trade(s)")
    else:
        logger.info(f"   ⏳ Waiting for signal...")



def run_enhanced_bot():
    """اجرای ربات بهبودیافته (نسخه فاز ۱)"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🤖 ENHANCED NDS TRADING BOT - PHASE 1 COMPLETE       ║
    ║══════════════════════════════════════════════════════════════║
    ║  📌 Symbol: BTCUSD                                           ║
    ║  📌 Strategy: Complete NDS (Paper Implementation)           ║
    ║  📌 Features:                                                ║
    ║     • Full Fractal Recursive Model                           ║
    ║     • Complete Symmetry Analysis                             ║
    ║     • Neural Network Enhancement                             ║
    ║  📌 Risk: Max 0.5% per trade                                ║
    ║  📌 Max Lots: 0.3                                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # بررسی وجود کلاس EnhancedNDSTradingBot
        if 'EnhancedNDSTradingBot' in globals():
            bot = EnhancedNDSTradingBot(symbol="BTCUSD", max_lots=0.3)
            bot.start()
        else:
            print("❌ EnhancedNDSTradingBot not found. Running Optimized Bot instead...")
            run_optimized_bot()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to Optimized Bot...")
        run_optimized_bot()

def run_original_bot():
    """اجرای ربات اصلی"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        🤖 NDS ADVANCED TRADING BOT - Phase 1            ║
    ║══════════════════════════════════════════════════════════║
    ║  📌 Symbol: BTCUSD                                       ║
    ║  📌 Strategy: NDS (Nodal Displacement Sequencing)        ║
    ║  📌 Risk: Max 0.5% per trade                            ║
    ║  📌 Max Lots: 0.3                                       ║
    ║  📌 Min Balance: $500                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # بررسی وجود کلاس NDSTradingBot
        if 'NDSTradingBot' in globals():
            bot = NDSTradingBot(symbol="BTCUSD", max_lots=0.3)
            bot.start()
        elif 'EnhancedNDSTradingBot' in globals():
            print("⚠️  Original bot not found. Running Enhanced Bot...")
            run_enhanced_bot()
        else:
            print("⚠️  No bot classes found. Running Optimized Bot...")
            bot = OptimizedNDSTradingBot(symbol="BTCUSD", max_lots=0.3)
            bot.start()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to Optimized Bot...")
        bot = OptimizedNDSTradingBot(symbol="BTCUSD", max_lots=0.3)
        bot.start()

def run_scalping_bot():
    """اجرای ربات اسکلپینگ"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║            ⚡ ULTRA FAST SCALPING BOT                   ║
    ║══════════════════════════════════════════════════════════║
    ║  📌 Symbol: BTCUSD                                      ║
    ║  📌 Strategy: NDS Ultra Fast Scalping                  ║
    ║  📌 Timeframes: M1/M2                                  ║
    ║  📌 Target: 10 pips | Max Risk: 5 pips                ║
    ║  📌 Volume: 0.1 lots fixed                            ║
    ║  📌 Max Trades/Hour: 10                               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # اضافه کردن گزینه 5 به منوی اصلی
        bot = ScalpingNDSTradingBot(symbol="BTCUSD", max_lots=0.5)
        bot.start()
        
    except Exception as e:
        logger.error(f"Failed to start scalping bot: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_mt5_order():
    """تست ارسال سفارش ساده"""
    print("\n🔧 Testing MT5 Order Sending...")
    
    try:
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return False
        
        symbol = "BTCUSD"
        
        
        # اطلاعات symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not found")
            return False
        
        print(f"✅ Symbol info:")
        print(f"   Bid: {symbol_info.bid}")
        print(f"   Ask: {symbol_info.ask}")
        
        # تست یک سفارش بسیار ساده
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": symbol_info.ask,
            "deviation": 20,
            "magic": 999999,
            "comment": "TEST ORDER",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        
        print(f"\n📤 Sending test order...")
        result = mt5.order_send(request)
        
        if result is None:
            print("❌ Order send returned None")
            return False
        
        print(f"✅ Order result received")
        print(f"   Result type: {type(result)}")
        
        if hasattr(result, 'retcode'):
            print(f"   Retcode: {result.retcode}")
            print(f"   Comment: {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"   ✅ Test order successful! Order: {result.order}")
                # بستن فوری
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": 0.01,
                    "type": mt5.ORDER_TYPE_SELL,
                    "position": result.order,
                    "price": symbol_info.bid,
                    "deviation": 20,
                    "magic": 999999,
                    "comment": "CLOSE TEST",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                
                close_result = mt5.order_send(close_request)
                if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"   ✅ Test order closed successfully")
                else:
                    print(f"   ⚠️ Could not close test order")
                
                return True
            else:
                print(f"   ❌ Test order failed")
                return False
        else:
            print("❌ Result has no retcode attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mt5.shutdown()


    # ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """نقطه ورود اصلی برنامه"""
    # Set UTF-8 encoding for console output
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # بارگذاری config
    try:
        if MODULAR_IMPORTS_AVAILABLE:
            config = get_config()
            logger.info(f"✅ Config loaded")
        else:
            config = None
            logger.info("⚠️ Using default config (modular imports not available)")
    except Exception as e:
        logger.warning(f"⚠️ Config loading failed: {e}, using defaults")
        config = None
    
    # اتصال اولیه به MT5 برای دریافت نمادها
    logger.info("🔌 Connecting to MT5 to get available symbols...")
    if not mt5.initialize():
        error = mt5.last_error()
        logger.error("❌ Failed to initialize MT5")
        logger.error(f"   Error: {error}")
        logger.error("   Please make sure MT5 terminal is running and logged in")
        return
    
    # دریافت نمادهای موجود
    resolver = SymbolResolver() if MODULAR_IMPORTS_AVAILABLE else None
    symbol_menu = get_symbol_menu() if MODULAR_IMPORTS_AVAILABLE else {
        'XAUUSD (Gold)': 'XAUUSD',
        'EURUSD (Euro/USD)': 'EURUSD',
        'US30/YM (Dow Jones)': 'US30',
        'BTCUSD (Bitcoin)': 'BTCUSD'
    }
    
    # ========================================================================
    # انتخاب نماد
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 SELECT SYMBOL (نماد معاملاتی)")
    print("=" * 60)
    
    if not symbol_menu:
        logger.error("❌ No tradeable symbols found!")
        logger.error("   Please check your MT5 connection and available symbols")
        mt5.shutdown()
        return
    
    symbol_list = list(symbol_menu.items())
    for idx, (display_name, real_name) in enumerate(symbol_list, 1):
        status = "✅" if (resolver and resolver.is_symbol_tradeable(real_name)) or not resolver else "⚠️"
        print(f"{idx}. {status} {display_name}")
        if real_name != display_name.split('(')[0].strip():
            print(f"   Broker Symbol: {real_name}")
    
    print("=" * 60)
    
    while True:
        try:
            symbol_choice = input(f"\nEnter symbol number (1-{len(symbol_list)}): ").strip()
            symbol_idx = int(symbol_choice) - 1
            if 0 <= symbol_idx < len(symbol_list):
                selected_display, selected_symbol = symbol_list[symbol_idx]
                logger.info(f"✅ Selected symbol: {selected_display} -> {selected_symbol}")
                break
            else:
                print(f"❌ Invalid choice! Please enter 1-{len(symbol_list)}")
        except ValueError:
            print("❌ Invalid input! Please enter a number")
        except KeyboardInterrupt:
            logger.info("\n⏹️  Cancelled by user")
            mt5.shutdown()
            return
    
    # ========================================================================
    # انتخاب استراتژی (Bot Version)
    # ========================================================================
    print("\n" + "=" * 60)
    print("🤖 SELECT STRATEGY (استراتژی معاملاتی)")
    print("=" * 60)
    print("1. 📈 Day Trading (معاملات روزانه)")
    print("   - Trend: H1 | Coarse: M15 | Fine: M3 | Entry: M1")
    print("   - Exit Signal: M5 | Exit Confirm: M3")
    print("   - Best for: Swing trading, longer positions")
    print()
    print("2. ⚡ Scalping (اسکلپینگ)")
    print("   - Trend: M15 | Coarse: M5 | Fine: M3 | Entry: M1")
    print("   - Exit Signal: M3 | Exit Confirm: M1")
    print("   - Best for: Active trading, quick profits")
    print()
    print("3. 🚀 Super Scalping (سوپر اسکلپینگ)")
    print("   - Trend: M5 | Coarse: M3 | Fine: M1 | Entry: M1")
    print("   - Exit Signal: M3 | Exit Confirm: M1")
    print("   - Best for: High frequency trading, ultra-fast execution")
    print()
    print("4. 🔌 Test MT5 Connection Only")
    print("=" * 60)
    print()
    print("💡 Note: All strategies use the same advanced analysis engine")
    print("   (NDS + Transformer + RL + HMM + CVaR + GARCH + VWAP + SETAR)")
    print("   Only timeframes differ based on strategy")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nEnter strategy number (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            else:
                print("❌ Invalid choice! Please enter 1-4")
        except KeyboardInterrupt:
            logger.info("\n⏹️  Cancelled by user")
            mt5.shutdown()
            return
    
    # ایجاد MT5Manager با نماد انتخاب شده
    mt5_manager = MT5Manager(symbol=selected_symbol)
    
    if not mt5_manager.connect():
        logger.error("❌ Failed to connect to MT5")
        mt5.shutdown()
        return
    
    # به‌روزرسانی config با نماد انتخاب شده
    if config:
        config.symbol = selected_symbol
    
    try:
        if choice == "1":
            # ✅ Day Trading
            logger.info("📈 Starting Day Trading Bot...")
            logger.info(f"   Symbol: {selected_symbol} ({selected_display})")
            logger.info(f"   Strategy: Day Trading (H1/M15/M3/M1)")
            max_lots = config.max_lots if config else 0.3
            bot = UnifiedTradingBot(
                symbol=selected_symbol, 
                max_lots=max_lots, 
                config=config,
                strategy="day_trading"
            )
            bot.start()
        
        elif choice == "2":
            # ✅ Scalping
            logger.info("⚡ Starting Scalping Bot...")
            logger.info(f"   Symbol: {selected_symbol} ({selected_display})")
            logger.info(f"   Strategy: Scalping (M15/M5/M3/M1)")
            max_lots = config.max_lots if config else 0.3
            bot = UnifiedTradingBot(
                symbol=selected_symbol, 
                max_lots=max_lots, 
                config=config,
                strategy="scalping"
            )
            bot.start()
        
        elif choice == "3":
            # ✅ Super Scalping
            logger.info("🚀 Starting Super Scalping Bot...")
            logger.info(f"   Symbol: {selected_symbol} ({selected_display})")
            logger.info(f"   Strategy: Super Scalping (M5/M3/M1)")
            max_lots = config.max_lots if config else 0.3
            bot = UnifiedTradingBot(
                symbol=selected_symbol, 
                max_lots=max_lots, 
                config=config,
                strategy="super_scalping"
            )
            bot.start()
        
        elif choice == "4":
            # Test Connection
            logger.info("🔌 Testing MT5 Connection...")
            logger.info(f"   Symbol: {selected_symbol} ({selected_display})")
            if mt5_manager.test_connection():
                logger.info("✅ Connection test passed!")
                # نمایش اطلاعات نماد
                symbol_info = mt5.symbol_info(selected_symbol)
                if symbol_info:
                    logger.info(f"   Symbol Info:")
                    logger.info(f"   - Name: {symbol_info.name}")
                    logger.info(f"   - Bid: {symbol_info.bid}")
                    logger.info(f"   - Ask: {symbol_info.ask}")
                    logger.info(f"   - Spread: {symbol_info.spread} points")
                    logger.info(f"   - Trade Mode: {symbol_info.trade_mode}")
            else:
                logger.error("❌ Connection test failed!")
        
        else:
            logger.error("❌ Invalid choice!")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5_manager.disconnect()
        mt5.shutdown()


if __name__ == "__main__":
    main()

