#!/usr/bin/env python3
"""
ULTIMATE TRADING BOT - SINGLE FILE 
Queue System: Max 5 Positions, 25 Trades/Hari
Mengambil API Keys dari Environment Variables/Secrets
"""

import os
import asyncio
import time
import requests
import json
import hmac
import hashlib
import base64
import logging
import sys
import traceback
from datetime import datetime
from collections import deque

# =============================================================================
# ⚙️ CONFIGURATION - AMAN (ambil dari environment)
# =============================================================================

class Config:
    # 🔐 API Keys dari Environment Variables/Secrets - AMAN!
    BITGET_API_KEY = os.getenv('BITGET_API_KEY')
    BITGET_SECRET_KEY = os.getenv('BITGET_SECRET_KEY')
    BITGET_PASSPHRASE = os.getenv('BITGET_PASSPHRASE')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # 💰 Trading Parameters
    INITIAL_CAPITAL = 2.0
    DRY_RUN = True  # ⚠️ SET False UNTUK LIVE TRADING!
    MIN_ORDER_SIZE = 1.2
    
    # 🎯 Queue System Parameters
    MAX_OPEN_POSITIONS = 5
    MAX_DAILY_TRADES = 25
    MIN_QUEUE_DISTANCE = 2  # minutes between trades
    
    # 🛡️ Risk Management
    STOP_LOSS_PCT = 0.015
    TAKE_PROFIT_PCT = 0.025
    MAX_DRAWDOWN = 0.30
    
    # Trading Universe
    COINS = ['HYPEUSDT', 'ASTERUSDT', 'SOLUSDT', 'AVAXUSDT', 'BTCUSDT', 'ETHUSDT']

# =============================================================================
# 🚀 BOT ENGINE - JANGAN EDIT DI BAWAH INI!
# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def send_message(self, message):
        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            payload = {'chat_id': Config.TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

class BitgetClient:
    def __init__(self):
        self.base_url = "https://api.bitget.com"
        self.telegram = TelegramBot()
        
    def _sign(self, timestamp, method, path, body=""):
        message = f"{timestamp}{method}{path}{body}"
        mac = hmac.new(
            bytes(Config.BITGET_SECRET_KEY, 'utf-8'), 
            bytes(message, 'utf-8'), 
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _headers(self, method, path, body=""):
        timestamp = str(int(time.time() * 1000))
        return {
            'ACCESS-KEY': Config.BITGET_API_KEY,
            'ACCESS-SIGN': self._sign(timestamp, method, path, body),
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': Config.BITGET_PASSPHRASE,
            'Content-Type': 'application/json'
        }
    
    def get_balance(self):
        """Get USDT futures balance"""
        try:
            method, path = 'GET', '/api/mix/v1/account/accounts?productType=usdt-futures'
            headers = self._headers(method, path)
            response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '00000' and data['data']:
                    return float(data['data'][0]['usdtEquity'])
            return Config.INITIAL_CAPITAL
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return Config.INITIAL_CAPITAL

    def get_kline_data(self, symbol, interval='1m', limit=100):
        """Get kline data without pandas"""
        try:
            method = 'GET'
            path = f'/api/mix/v1/market/candles?symbol={symbol}&granularity={interval}&limit={limit}'
            headers = self._headers(method, path)
            response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '00000':
                    return self._parse_candle_data(data['data'])
            return None
        except Exception as e:
            logger.error(f"Kline error {symbol}: {e}")
            return None

    def _parse_candle_data(self, candle_data):
        """Parse candle data to simple format"""
        candles = []
        for candle in candle_data:
            try:
                candles.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            except (ValueError, IndexError):
                continue
        return candles

    def place_order(self, symbol, side, size, leverage=10, stop_loss=None, take_profit=None):
        """Place futures order"""
        if Config.DRY_RUN:
            logger.info(f"DRY RUN: {side} {size} {symbol} at {leverage}x")
            return {'order_id': 'dry_run'}
            
        try:
            method, path = 'POST', '/api/mix/v1/order/placeOrder'
            
            order_data = {
                "symbol": symbol,
                "productType": "usdt-futures",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "size": str(round(size, 3)),
                "side": side,
                "orderType": "market",
                "leverage": str(leverage)
            }
            
            if stop_loss:
                order_data["presetStopLossPrice"] = str(round(stop_loss, 6))
            if take_profit:
                order_data["presetTakeProfitPrice"] = str(round(take_profit, 6))
            
            body = json.dumps(order_data)
            headers = self._headers(method, path, body)
            
            response = requests.post(
                f"{self.base_url}{path}", 
                headers=headers, 
                data=body,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['code'] == '00000':
                    logger.info(f"✅ ORDER EXECUTED: {side} {size} {symbol} at {leverage}x")
                    return result['data']
                else:
                    logger.error(f"Order failed: {result['msg']}")
            return None
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

class TechnicalCalculator:
    """Simple technical indicators without pandas"""
    
    @staticmethod
    def calculate_ema(prices, period):
        """Calculate EMA manually"""
        if len(prices) < period:
            return None
            
        ema = prices[0]
        multiplier = 2 / (period + 1)
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return ema
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI manually"""
        if len(prices) < period + 1:
            return 50
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def get_rolling_max(prices, window):
        """Get rolling maximum"""
        if len(prices) < window:
            return max(prices) if prices else 0
        return max(prices[-window:])
    
    @staticmethod
    def get_rolling_min(prices, window):
        """Get rolling minimum"""
        if len(prices) < window:
            return min(prices) if prices else 0
        return min(prices[-window:])

class StrategyEngine:
    def __init__(self):
        self.calculator = TechnicalCalculator()
        self.leverage_map = {
            'BTCUSDT': 20, 'ETHUSDT': 20, 
            'SOLUSDT': 15, 'AVAXUSDT': 15,
            'HYPEUSDT': 15, 'ASTERUSDT': 15,
            'DOGEUSDT': 8, 'SHIBUSDT': 8
        }
    
    def get_leverage(self, symbol):
        return self.leverage_map.get(symbol, 10)

    def ema_strategy(self, candles):
        """EMA trend strategy"""
        try:
            if len(candles) < 50:
                return 'NEUTRAL', 0.0
                
            closes = [c['close'] for c in candles]
            current_price = closes[-1]
            current_candle = candles[-1]
            
            # Calculate EMAs
            ema_9 = self.calculator.calculate_ema(closes, 9)
            ema_21 = self.calculator.calculate_ema(closes, 21)
            ema_50 = self.calculator.calculate_ema(closes, 50)
            
            if not all([ema_9, ema_21, ema_50]):
                return 'NEUTRAL', 0.0
            
            # Bullish EMA stack
            if (current_price > ema_9 > ema_21 > ema_50 and
                current_candle['close'] > current_candle['open']):
                return 'LONG', 0.82
                
            # Bearish EMA stack  
            if (current_price < ema_9 < ema_21 < ema_50 and
                current_candle['close'] < current_candle['open']):
                return 'SHORT', 0.82
                
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"EMA strategy error: {e}")
            return 'NEUTRAL', 0.0

    def breakout_strategy(self, candles):
        """Breakout strategy with volume"""
        try:
            if len(candles) < 20:
                return 'NEUTRAL', 0.0
                
            current_candle = candles[-1]
            current_price = current_candle['close']
            previous_candles = candles[:-1]
            
            highs = [c['high'] for c in previous_candles]
            lows = [c['low'] for c in previous_candles]
            volumes = [c['volume'] for c in previous_candles]
            
            high_20 = self.calculator.get_rolling_max(highs, 20)
            low_20 = self.calculator.get_rolling_min(lows, 20)
            volume_avg = sum(volumes[-20:]) / min(len(volumes), 20)
            
            current_volume = current_candle['volume']
            
            # Bullish breakout
            if (current_price > high_20 and 
                current_volume > volume_avg * 1.8):
                return 'LONG', 0.78
                
            # Bearish breakout
            if (current_price < low_20 and
                current_volume > volume_avg * 1.8):
                return 'SHORT', 0.78
                
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Breakout strategy error: {e}")
            return 'NEUTRAL', 0.0

    def generate_signal(self, symbol, market_data):
        """Generate trading signal"""
        try:
            # Get data from different timeframes
            data_5m = market_data.get('5m', [])
            data_15m = market_data.get('15m', [])
            
            if not data_5m or not data_15m:
                return 'NEUTRAL', 0.0
            
            # Run strategies
            strategies = {
                'ema': self.ema_strategy(data_15m),
                'breakout': self.breakout_strategy(data_5m),
            }
            
            # Calculate confidence
            long_confidence = 0
            short_confidence = 0
            total_strategies = 0
            
            for signal, confidence in strategies.values():
                if signal == 'LONG':
                    long_confidence += confidence
                    total_strategies += 1
                elif signal == 'SHORT':
                    short_confidence += confidence
                    total_strategies += 1
            
            # Require at least 2 strategies agreement
            if total_strategies >= 2:
                if long_confidence >= 1.5:  # 75% average confidence
                    return 'LONG', long_confidence / total_strategies
                elif short_confidence >= 1.5:
                    return 'SHORT', short_confidence / total_strategies
            
            return 'NEUTRAL', 0.0
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return 'NEUTRAL', 0.0

class QueueTradingSystem:
    def __init__(self):
        self.open_positions = []
        self.pending_signals = []
        self.trade_history = []
        self.last_trade_time = 0
        
        self.daily_stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'winning_trades': 0
        }
    
    def can_open_new_trade(self):
        """Check if we can open new trade"""
        # Max open positions
        if len(self.open_positions) >= Config.MAX_OPEN_POSITIONS:
            return False
            
        # Max daily trades
        if self.daily_stats['trades_opened'] >= Config.MAX_DAILY_TRADES:
            return False
            
        # Rate limiting
        time_since_last = time.time() - self.last_trade_time
        if time_since_last < Config.MIN_QUEUE_DISTANCE * 60:
            return False
            
        return True
    
    def add_signal_to_queue(self, symbol, signal, confidence):
        """Add signal to queue"""
        signal_data = {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'timestamp': time.time(),
            'priority': confidence
        }
        
        self.pending_signals.append(signal_data)
        # Sort by priority (highest confidence first)
        self.pending_signals.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"📥 Signal queued: {symbol} {signal} ({confidence:.1%})")
    
    def process_queue(self):
        """Process next signal from queue"""
        if not self.pending_signals or not self.can_open_new_trade():
            return None
            
        # Take highest priority signal
        signal_data = self.pending_signals.pop(0)
        logger.info(f"📤 Processing queue: {signal_data['symbol']} {signal_data['signal']}")
        
        return signal_data
    
    def open_position(self, symbol, signal, size, leverage, entry_price):
        """Record new position"""
        position = {
            'symbol': symbol,
            'signal': signal,
            'size': size,
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'status': 'OPEN'
        }
        
        self.open_positions.append(position)
        self.daily_stats['trades_opened'] += 1
        self.last_trade_time = time.time()
        
        logger.info(f"📍 Position opened: {symbol} {signal} ${size:.2f} - Open: {len(self.open_positions)}/5")
    
    def close_position(self, symbol, pnl=0):
        """Close position"""
        for i, pos in enumerate(self.open_positions):
            if pos['symbol'] == symbol:
                closed_pos = self.open_positions.pop(i)
                closed_pos['exit_time'] = datetime.now()
                closed_pos['pnl'] = pnl
                closed_pos['status'] = 'CLOSED'
                
                self.trade_history.append(closed_pos)
                self.daily_stats['trades_closed'] += 1
                self.daily_stats['total_pnl'] += pnl
                if pnl > 0:
                    self.daily_stats['winning_trades'] += 1
                
                logger.info(f"✅ Position closed: {symbol} PnL: ${pnl:.3f}")
                return True
        return False
    
    def get_queue_status(self):
        """Get current status"""
        return {
            'open_positions': len(self.open_positions),
            'max_positions': Config.MAX_OPEN_POSITIONS,
            'pending_signals': len(self.pending_signals),
            'daily_trades': self.daily_stats['trades_opened'],
            'max_daily_trades': Config.MAX_DAILY_TRADES
        }

class CapitalManager:
    def __init__(self):
        self.current_capital = Config.INITIAL_CAPITAL
        
    def get_position_size(self, confidence):
        """Calculate position size"""
        if self.current_capital < 10:
            base_size = self.current_capital * 0.40  # Aggressive
        else:
            base_size = self.current_capital * 0.25  # Moderate
            
        # Confidence adjustment
        size = base_size * (0.5 + confidence * 0.5)
        return max(size, Config.MIN_ORDER_SIZE)
    
    def update_capital(self, pnl):
        """Update capital"""
        self.current_capital += pnl
        
    def can_trade(self):
        """Check if we can trade"""
        return self.current_capital >= Config.MIN_ORDER_SIZE

class UltimateTradingBot:
    def __init__(self):
        self.client = BitgetClient()
        self.strategy = StrategyEngine()
        self.queue_system = QueueTradingSystem()
        self.capital_manager = CapitalManager()
        self.telegram = TelegramBot()
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("🚀 Ultimate Trading Bot Started - Queue System Active")
        
    async def run(self):
        """Main trading loop"""
        # Startup message
        self.telegram.send_message(
            f"🤖 <b>BOT STARTED</b>\n"
            f"💰 Capital: ${Config.INITIAL_CAPITAL:.2f}\n"
            f"🎯 Queue: {Config.MAX_OPEN_POSITIONS} max positions\n"
            f"📊 Daily: {Config.MAX_DAILY_TRADES} max trades\n"
            f"⚡ Mode: {'DRY RUN' if Config.DRY_RUN else 'LIVE'}"
        )
        
        while self.is_running:
            try:
                # Update balance
                balance = self.client.get_balance()
                if balance > 0:
                    self.capital_manager.current_capital = balance
                
                # Main workflow
                await self.scan_signals()
                await self.process_trade_queue()
                await self.manage_positions()
                
                # Status update
                if int(time.time()) % 300 < 30:  # Every 5 minutes
                    await self.send_status_update()
                
                await asyncio.sleep(30)  # 30 seconds between cycles
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def scan_signals(self):
        """Scan for trading signals"""
        for symbol in Config.COINS:
            try:
                # Get multi-timeframe data
                market_data = {
                    '5m': self.client.get_kline_data(symbol, '5m', 50),
                    '15m': self.client.get_kline_data(symbol, '15m', 50)
                }
                
                if not market_data['5m'] or not market_data['15m']:
                    continue
                    
                # Generate signal
                signal, confidence = self.strategy.generate_signal(symbol, market_data)
                
                # Add high confidence signals to queu
                if signal != 'NEUTRAL' and confidence >= 0.75:
                    self.queue_system.add_signal_to_queue(symbol, signal, confidence)
                    
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
    
    async def process_trade_queue(self):
        """Process trade queue"""
        if not self.capital_manager.can_trade():
            return
            
        signal_data = self.queue_system.process_queue()
        if not signal_data:
            return
            
        symbol, signal, confidence = signal_data['symbol'], signal_data['signal'], signal_data['confidence']
        
        try:
            # Get current data
            market_data = self.client.get_kline_data(symbol, '5m', 10)
            if not market_data:
                return
                
            current_price = market_data[-1]['close']
            leverage = self.strategy.get_leverage(symbol)
            position_size = self.capital_manager.get_position_size(confidence)
            
            # Calculate SL/TP
            if signal == 'LONG':
                stop_loss = current_price * (1 - Config.STOP_LOSS_PCT)
                take_profit = current_price * (1 + Config.TAKE_PROFIT_PCT)
                side = 'open_long'
            else:
                stop_loss = current_price * (1 + Config.STOP_LOSS_PCT)
                take_profit = current_price * (1 - Config.TAKE_PROFIT_PCT)
                side = 'open_short'
            
            # Execute order
            order_result = self.client.place_order(
                symbol=symbol,
                side=side,
                size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_result:
                self.queue_system.open_position(symbol, signal, position_size, leverage, current_price)
                
                # Send alert
                self.telegram.send_message(
                    f"🎯 <b>TRADE EXECUTED</b>\n"
                    f"💰 {symbol} {signal}\n"
                    f"📊 Size: ${position_size:.2f}\n"
                    f"🎪 Leverage: {leverage}x\n"
                    f"📈 Open: {len(self.queue_system.open_positions)}/5"
                )
                
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
    
    async def manage_positions(self):
        """Manage open positions"""
        # In live trading, this would check actual position status
        # For dry run, simulate some management
        if Config.DRY_RUN and self.queue_system.open_positions:
            import random
            if random.random() < 0.1:  # 10% chance to close a position
                pos = random.choice(self.queue_system.open_positions)
                pnl = random.choice([0.02, -0.01]) * pos['size']  # +2% or -1%
                self.queue_system.close_position(pos['symbol'], pnl)
                self.capital_manager.update_capital(pnl)
    
    async def send_status_update(self):
        """Send status update"""
        status = self.queue_system.get_queue_status()
        msg = f"""
📊 <b>QUEUE STATUS</b>
────────────────────
📍 Open Positions: {status['open_positions']}/{status['max_positions']}
📥 Queue Signals: {status['pending_signals']}
🎯 Daily Trades: {status['daily_trades']}/{status['max_daily_trades']}
💰 Capital: ${self.capital_manager.current_capital:.2f}
⏰ Runtime: {(datetime.now() - self.start_time).seconds // 60}m
────────────────────
        """
        self.telegram.send_message(msg)

class AutoRestart:
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 5
        
    async def run_with_restart(self):
        """Run with auto-restart"""
        while self.restart_count < self.max_restarts:
            try:
                logger.info(f"🔄 Starting bot (attempt {self.restart_count + 1})")
                bot = UltimateTradingBot()
                await bot.run()
            except Exception as e:
                self.restart_count += 1
                logger.error(f"Bot crashed: {e}")
                if self.restart_count < self.max_restarts:
                    logger.info("Waiting 30 seconds before restart...")
                    await asyncio.sleep(30)

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

async def main():
    print("🚀 Ultimate Trading Bot - Single File")
    print("=" * 50)
    
    # Check API keys
    required_vars = ['BITGET_API_KEY', 'BITGET_SECRET_KEY', 'BITGET_PASSPHRASE']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ ERROR: Missing environment variables: {', '.join(missing)}")
        print("\n💡 Cara setup:")
        print("Google Colab:")
        print("   os.environ['BITGET_API_KEY'] = 'your_key_here'")
        print("\nTerminal:")
        print("   export BITGET_API_KEY=your_key_here")
        print("\nGitHub Secrets:")
        print("   Set in Repository Settings → Secrets and variables → Actions")
        return
    
    print("✅ API keys verified")
    print("🤖 Starting Ultimate Trading Bot...")
    
    # Run with auto-restart
    restart_manager = AutoRestart()
    await restart_manager.run_with_restart()

if __name__ == "__main__":
    # Check if running in interactive mode (Google Colab)
    try:
        import google.colab
        print("🔍 Detected Google Colab environment")
        print("💡 Run this in a cell before the bot:")
        print("""
import os
os.environ['BITGET_API_KEY'] = 'your_api_key_here'
os.environ['BITGET_SECRET_KEY'] = 'your_secret_here'  
os.environ['BITGET_PASSPHRASE'] = 'your_passphrase_here'
        """)
    except ImportError:
        pass  # Not in Colab
    
    # Run the bot
    asyncio.run(main())
