"""
Symbol Resolver - حل کننده نام نمادها
این ماژول نام نمادها را با نام واقعی در بروکر تطبیق می‌دهد
"""
import MetaTrader5 as mt5
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SymbolResolver:
    """کلاس برای پیدا کردن نام واقعی نمادها در بروکر"""
    
    # نمادهای استاندارد و نام‌های جایگزین احتمالی
    SYMBOL_PATTERNS = {
        'XAUUSD': ['XAUUSD', 'GOLD', 'GOLDUSD', 'XAU/USD', 'GOLD/USD', 'XAUUSD.', 'GOLD.'],
        'EURUSD': ['EURUSD', 'EUR/USD', 'EURUSD.', 'EUR/USD.'],
        'US30': ['US30', 'YM', 'YM.', 'US30.', 'DOW', 'DOW.', 'US30USD', 'YMUSD'],
        'BTCUSD': ['BTCUSD', 'BTC/USD', 'BTCUSD.', 'BTC/USD.', 'BITCOIN', 'BITCOINUSD']
    }
    
    def __init__(self):
        self.available_symbols: List[str] = []
        self.symbol_map: Dict[str, str] = {}  # استاندارد -> واقعی
    
    def get_available_symbols(self) -> List[str]:
        """دریافت لیست تمام نمادهای موجود در بروکر"""
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.warning("⚠️ No symbols available from MT5")
                return []
            
            self.available_symbols = [s.name for s in symbols if s.visible]
            logger.info(f"✅ Found {len(self.available_symbols)} available symbols")
            return self.available_symbols
        except Exception as e:
            logger.error(f"❌ Error getting symbols: {e}")
            return []
    
    def find_symbol(self, standard_name: str) -> Optional[str]:
        """
        پیدا کردن نام واقعی نماد در بروکر
        
        Args:
            standard_name: نام استاندارد نماد (مثلاً 'XAUUSD')
        
        Returns:
            نام واقعی نماد در بروکر یا None
        """
        if not self.available_symbols:
            self.get_available_symbols()
        
        # اگر قبلاً پیدا شده، برگردان
        if standard_name.upper() in self.symbol_map:
            return self.symbol_map[standard_name.upper()]
        
        # الگوهای احتمالی
        patterns = self.SYMBOL_PATTERNS.get(standard_name.upper(), [standard_name.upper()])
        
        # جستجوی دقیق
        for pattern in patterns:
            # جستجوی دقیق
            if pattern in self.available_symbols:
                # بررسی tradeable بودن
                if self.is_symbol_tradeable(pattern):
                    self.symbol_map[standard_name.upper()] = pattern
                    logger.info(f"✅ Found {standard_name} as {pattern}")
                    return pattern
            
            # جستجوی case-insensitive
            for symbol in self.available_symbols:
                if symbol.upper() == pattern.upper():
                    if self.is_symbol_tradeable(symbol):
                        self.symbol_map[standard_name.upper()] = symbol
                        logger.info(f"✅ Found {standard_name} as {symbol}")
                        return symbol
        
        # جستجوی partial match (برای پسوند/پیشوند)
        # اولویت با نمادهایی که دقیقاً شامل الگو هستند
        best_match = None
        best_score = 0
        
        for pattern in patterns:
            pattern_upper = pattern.upper().replace('/', '').replace('.', '')
            for symbol in self.available_symbols:
                symbol_upper = symbol.upper().replace('/', '').replace('.', '')
                
                # امتیازدهی برای تطابق
                score = 0
                if pattern_upper == symbol_upper:
                    score = 100  # تطابق کامل
                elif pattern_upper in symbol_upper:
                    score = 80 - (len(symbol_upper) - len(pattern_upper))  # هر چه کوتاه‌تر بهتر
                elif symbol_upper in pattern_upper:
                    score = 60
                
                if score > best_score and self.is_symbol_tradeable(symbol):
                    best_score = score
                    best_match = symbol
        
        if best_match:
            self.symbol_map[standard_name.upper()] = best_match
            logger.info(f"✅ Found {standard_name} as {best_match} (best match, score: {best_score})")
            return best_match
        
        logger.warning(f"⚠️ Symbol {standard_name} not found in broker")
        return None
    
    def get_all_available_standard_symbols(self) -> Dict[str, str]:
        """
        دریافت تمام نمادهای استاندارد که در بروکر موجود هستند
        
        Returns:
            Dict با کلید نام استاندارد و مقدار نام واقعی
        """
        result = {}
        for standard_name in self.SYMBOL_PATTERNS.keys():
            real_name = self.find_symbol(standard_name)
            if real_name:
                result[standard_name] = real_name
        
        return result
    
    def is_symbol_tradeable(self, symbol: str) -> bool:
        """بررسی اینکه نماد قابل معامله است یا نه"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # بررسی visibility و trade mode
            return (symbol_info.visible and 
                   symbol_info.trade_mode != 0 and
                   symbol_info.trade_mode != 2)  # 0 = disabled, 2 = close only
        except Exception as e:
            logger.error(f"❌ Error checking symbol {symbol}: {e}")
            return False
    
    def get_symbol_display_name(self, standard_name: str, real_name: str) -> str:
        """نام نمایشی برای نماد"""
        if standard_name.upper() == real_name.upper():
            return standard_name
        return f"{standard_name} ({real_name})"


def get_symbol_menu() -> Dict[str, str]:
    """
    ایجاد منوی انتخاب نماد
    
    Returns:
        Dict با کلید نام نمایشی و مقدار نام واقعی نماد
    """
    resolver = SymbolResolver()
    available = resolver.get_all_available_standard_symbols()
    
    menu = {}
    display_names = {
        'XAUUSD': 'XAUUSD (Gold)',
        'EURUSD': 'EURUSD (Euro/USD)',
        'US30': 'US30/YM (Dow Jones)',
        'BTCUSD': 'BTCUSD (Bitcoin)'
    }
    
    for standard_name, real_name in available.items():
        if resolver.is_symbol_tradeable(real_name):
            display = display_names.get(standard_name, standard_name)
            menu[display] = real_name
    
    return menu

