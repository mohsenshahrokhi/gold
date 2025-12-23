# 📦 راهنمای Modularization

## 🎯 هدف
این سند راهنمای تقسیم کد به ماژول‌های مناسب است.

## 📁 ساختار پیشنهادی

```
trading_bot/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── config.py          # BotConfig, RiskConfig, etc.
│   └── constants.py       # تمام constants
├── core/
│   ├── __init__.py
│   ├── interfaces.py      # Protocols و Interfaces
│   ├── exceptions.py      # Custom exceptions
│   └── types.py           # Dataclasses (Node, Cycle, NDSSignal, etc.)
├── managers/
│   ├── __init__.py
│   ├── mt5_manager.py     # MT5Manager
│   ├── risk_manager.py    # RiskManager
│   └── trade_manager.py   # TradeManager
├── analyzers/
│   ├── __init__.py
│   ├── base_analyzer.py   # AdvancedNDSAnalyzer
│   ├── enhanced_analyzer.py  # EnhancedNDSAnalyzer
│   ├── optimized_analyzer.py # OptimizedNDSAnalyzer
│   └── models/            # مدل‌های مقاله
│       ├── transformer.py
│       ├── gnn.py
│       ├── rl.py
│       ├── hmm.py
│       ├── cvar.py
│       ├── garch.py
│       ├── vwap.py
│       └── setar.py
├── bots/
│   ├── __init__.py
│   ├── base_bot.py        # NDSTradingBot
│   ├── enhanced_bot.py    # EnhancedNDSTradingBot
│   ├── optimized_bot.py   # OptimizedNDSTradingBot
│   ├── scalping_bot.py    # ScalpingNDSTradingBot
│   └── professional_scalping_bot.py
├── strategies/
│   ├── __init__.py
│   ├── nds_strategy.py
│   ├── scalping_strategy.py
│   └── trailing/
│       ├── node_based_trailing.py
│       └── improved_trailing.py
└── utils/
    ├── __init__.py
    ├── cache.py
    └── performance.py
```

## ✅ کارهای انجام شده برای آماده‌سازی

### 1. Interfaces ایجاد شد
- `interfaces.py` با Protocols برای decoupling
- آماده برای dependency injection

### 2. Configuration Management
- `config.py` با BotConfig و nested configs
- `constants.py` برای تمام magic numbers
- پشتیبانی از JSON config files

### 3. Custom Exceptions
- `exceptions.py` با exception hierarchy
- Error handling بهتر

### 4. Dependency Injection Ready
- کلاس‌ها config را به عنوان parameter می‌گیرند
- Fallback به defaults اگر config موجود نباشد

## 🔄 مراحل Modularization

### مرحله 1: Extract Types
```python
# types.py
from dataclasses import dataclass
from enum import Enum
# تمام dataclasses و enums
```

### مرحله 2: Extract Managers
```python
# managers/mt5_manager.py
from ..core.interfaces import IMT5Manager
from ..core.exceptions import MT5ConnectionError
# MT5Manager class
```

### مرحله 3: Extract Analyzers
```python
# analyzers/base_analyzer.py
from ..managers.mt5_manager import MT5Manager
from ..config.config import BotConfig
# AdvancedNDSAnalyzer
```

### مرحله 4: Extract Bots
```python
# bots/base_bot.py
from ..managers import MT5Manager, RiskManager, TradeManager
from ..analyzers import AdvancedNDSAnalyzer
# NDSTradingBot
```

## 🧪 Testing Strategy

### Unit Tests
- هر ماژول تست مستقل داشته باشد
- Mock dependencies با interfaces

### Integration Tests
- تست ارتباط بین ماژول‌ها
- تست با config واقعی

## 📝 نکات مهم

1. **Circular Dependencies**: از interfaces استفاده کنید
2. **Import Paths**: از relative imports استفاده کنید
3. **Backward Compatibility**: کد فعلی باید کار کند
4. **Gradual Migration**: می‌توانید تدریجی modularize کنید

