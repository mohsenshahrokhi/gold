# راهنمای تنظیم Environment Variables

## تنظیم برای ربات تلگرام

### روش 1: استفاده از فایل‌های Setup (ساده‌ترین)

**Windows Command Prompt:**
1. فایل `setup_telegram.bat` را باز کنید
2. توکن و Chat ID را وارد کنید
3. فایل را ذخیره کنید
4. قبل از اجرای ربات، این فایل را اجرا کنید:
   ```cmd
   setup_telegram.bat
   ```

**Windows PowerShell:**
1. فایل `setup_telegram.ps1` را باز کنید
2. توکن و Chat ID را وارد کنید
3. فایل را ذخیره کنید
4. قبل از اجرای ربات، این فایل را اجرا کنید:
   ```powershell
   .\setup_telegram.ps1
   ```

### روش 2: تنظیم دستی (دائمی)

**Windows (Command Prompt):**
```cmd
setx TELEGRAM_BOT_TOKEN "8587645622:AAENtRBRFlTf831OI2zJcDT3dbJU-jpBSsc"
setx TELEGRAM_CHAT_ID "@GoldMt5AlgoBot"
```

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "8587645622:AAENtRBRFlTf831OI2zJcDT3dbJU-jpBSsc", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "@GoldMt5AlgoBot", "User")
```

**Linux/Mac:**
```bash
export TELEGRAM_BOT_TOKEN="8587645622:AAENtRBRFlTf831OI2zJcDT3dbJU-jpBSsc"
export TELEGRAM_CHAT_ID="@GoldMt5AlgoBot"
```

برای دائمی کردن در Linux/Mac، به فایل `~/.bashrc` یا `~/.zshrc` اضافه کنید.

### روش 3: تنظیم موقت (فقط برای این Session)

**Windows (Command Prompt):**
```cmd
set TELEGRAM_BOT_TOKEN=8587645622:AAENtRBRFlTf831OI2zJcDT3dbJU-jpBSsc
set TELEGRAM_CHAT_ID=@GoldMt5AlgoBot
```

**Windows (PowerShell):**
```powershell
$env:TELEGRAM_BOT_TOKEN="8587645622:AAENtRBRFlTf831OI2zJcDT3dbJU-jpBSsc"
$env:TELEGRAM_CHAT_ID="@GoldMt5AlgoBot"
```

## تست

بعد از تنظیم، ربات را اجرا کنید. در لاگ باید ببینید:
```
✅ Telegram notifications enabled
```

## امنیت

⚠️ **مهم:** توکن ربات را در GitHub قرار ندهید!
- فایل‌های `setup_telegram.bat` و `setup_telegram.ps1` را با توکن واقعی خود پر کنید
- این فایل‌ها در `.gitignore` نیستند، پس توکن را در آن‌ها قرار ندهید
- از Environment Variables استفاده کنید

