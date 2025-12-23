# راهنمای تنظیم ربات تلگرام

## مرحله 1: ساخت ربات تلگرام و دریافت توکن

1. در تلگرام به **@BotFather** پیام دهید
2. دستور `/newbot` را ارسال کنید
3. نام ربات را وارد کنید (مثلاً: `My Trading Bot`)
4. Username ربات را وارد کنید (باید به `bot` ختم شود، مثلاً: `my_trading_bot`)
5. BotFather یک **توکن** به شما می‌دهد که شبیه این است:
   ```
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
   این همان **TELEGRAM_BOT_TOKEN** است

## مرحله 2: دریافت Chat ID

### روش 1: استفاده از @userinfobot
1. به ربات **@userinfobot** پیام دهید
2. Chat ID شما نمایش داده می‌شود (مثلاً: `123456789`)

### روش 2: استفاده از API
1. به ربات خود پیام دهید (مثلاً `/start`)
2. به این آدرس بروید (توکن خود را جایگزین کنید):
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. در پاسخ JSON، `chat.id` را پیدا کنید
4. این عدد همان **TELEGRAM_CHAT_ID** است

## مرحله 3: تنظیم در کد

### روش 1: Environment Variables (پیشنهادی)

**Windows (Command Prompt):**
```cmd
set TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
set TELEGRAM_CHAT_ID=123456789
```

**Windows (PowerShell):**
```powershell
$env:TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
$env:TELEGRAM_CHAT_ID="123456789"
```

**Linux/Mac:**
```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"
```

### روش 2: فایل config.json

فایل `config.json` را باز کنید و این بخش را اضافه/ویرایش کنید:

```json
{
  "telegram": {
    "bot_token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
    "chat_id": "123456789"
  }
}
```

### روش 3: فایل config.py

اگر از config.py استفاده می‌کنید، می‌توانید مستقیماً در کد تنظیم کنید:

```python
config.telegram_token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
config.telegram_chat_id = "123456789"
```

## اولویت استفاده

کد به ترتیب زیر جستجو می‌کند:
1. Environment Variables (`TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID`)
2. Config object (`config.telegram_token` و `config.telegram_chat_id`)
3. Config dict (`config.telegram['bot_token']` و `config.telegram['chat_id']`)

## تست

بعد از تنظیم، ربات را اجرا کنید. اگر تنظیمات درست باشد، در لاگ می‌بینید:
```
✅ Telegram notifications enabled
```

اگر تنظیم نشده باشد:
```
⚠️ Telegram notifications disabled (no token/chat_id)
```

## نکات امنیتی

⚠️ **مهم:** توکن ربات را در GitHub یا جاهای عمومی قرار ندهید!
- از Environment Variables استفاده کنید
- یا فایل `config.json` را به `.gitignore` اضافه کنید

