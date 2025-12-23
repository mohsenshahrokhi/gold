"""
Configuration Management
مدیریت تنظیمات مرکزی برای ربات معاملاتی
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
import json
import os
from pathlib import Path


@dataclass
class RiskConfig:
    """تنظیمات مدیریت ریسک"""
    max_risk_percent: float = 0.5
    max_lots: float = 0.3
    min_balance: float = 500.0
    max_daily_loss: float = 0.02
    max_daily_trades: int = 10


@dataclass
class TradingConfig:
    """تنظیمات معاملاتی"""
    magic_number: int = 123456
    pip_margin: int = 5
    min_volume: float = 0.01
    max_volume: float = 0.5
    spread_margin: float = 5.0
    commission: float = 0.0


@dataclass
class ScalpingConfig:
    """تنظیمات اسکلپینگ"""
    max_trades_per_hour: int = 10
    target_pips: int = 10
    max_risk_pips: int = 5
    min_confidence: float = 0.7
    fixed_volume: float = 0.1


@dataclass
class NDSConfig:
    """تنظیمات NDS"""
    alpha_correction: float = 0.86
    alpha_pressure: float = 0.2
    beta_displacement: float = 0.3
    tf_trend: int = 3  # M3
    tf_analysis: int = 1  # M1
    tf_entry: int = 1  # M1


@dataclass
class BotConfig:
    """تنظیمات کلی ربات"""
    symbol: str = "BTCUSD"
    max_lots: float = 0.3
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    scalping: ScalpingConfig = field(default_factory=ScalpingConfig)
    nds: NDSConfig = field(default_factory=NDSConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 60
    
    @classmethod
    def from_file(cls, path: str) -> 'BotConfig':
        """بارگذاری تنظیمات از فایل JSON"""
        if not os.path.exists(path):
            # Create default config
            default = cls()
            default.to_file(path)
            return default
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        if 'risk' in data:
            data['risk'] = RiskConfig(**data['risk'])
        if 'trading' in data:
            data['trading'] = TradingConfig(**data['trading'])
        if 'scalping' in data:
            data['scalping'] = ScalpingConfig(**data['scalping'])
        if 'nds' in data:
            data['nds'] = NDSConfig(**data['nds'])
        
        return cls(**data)
    
    def to_file(self, path: str):
        """ذخیره تنظیمات در فایل JSON"""
        # Convert to dict with nested dataclasses
        data = asdict(self)
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def update(self, **kwargs):
        """به‌روزرسانی تنظیمات"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (RiskConfig, TradingConfig, ScalpingConfig, NDSConfig)):
                    # Update nested config
                    nested = getattr(self, key)
                    for k, v in value.items():
                        if hasattr(nested, k):
                            setattr(nested, k, v)
                else:
                    setattr(self, key, value)


# Global config instance
_config: Optional[BotConfig] = None


def get_config() -> BotConfig:
    """Get global config instance"""
    global _config
    if _config is None:
        config_path = os.getenv('BOT_CONFIG_PATH', 'config.json')
        _config = BotConfig.from_file(config_path)
    return _config


def set_config(config: BotConfig):
    """Set global config instance"""
    global _config
    _config = config

