#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run the trading bot with proper encoding
"""
import sys
import io

# Set UTF-8 encoding for console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import and run main
if __name__ == "__main__":
    try:
        # Import the main module
        import importlib.util
        spec = importlib.util.spec_from_file_location("sonnet_bot", "sonnet copy 7.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run main
        module.main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

