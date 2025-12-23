# تنظیم Environment Variables برای ربات تلگرام (PowerShell)
# این فایل را قبل از اجرای ربات اجرا کنید

$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
$env:TELEGRAM_CHAT_ID = "@YOUR_CHANNEL_NAME"

Write-Host "Telegram variables set!"
Write-Host "Bot Token: $env:TELEGRAM_BOT_TOKEN"
Write-Host "Chat ID: $env:TELEGRAM_CHAT_ID"
Write-Host ""
Write-Host "Now you can run the bot..."

