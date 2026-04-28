"""
Polymarket WebSocket Price Monitor

Real-time price updates via WebSocket instead of polling.
Subscribes to token IDs for open positions and updates prices live.

Usage:
    from src.Polymarket.websocket_monitor import WebSocketPriceMonitor

    monitor = WebSocketPriceMonitor(on_price_update=callback)
    monitor.subscribe(["token_id_1", "token_id_2"])
    monitor.start()
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set
from pathlib import Path

try:
    import websocket
except ImportError:
    websocket = None
    print("WARNING: websocket-client not installed. Run: pip install websocket-client")

# WebSocket endpoints
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Heartbeat interval (seconds)
PING_INTERVAL = 10

# Data directory
DATA_DIR = Path("Data/paper_trading")
POSITIONS_FILE = DATA_DIR / "positions.json"


class WebSocketPriceMonitor:
    """
    WebSocket-based price monitor for Polymarket.

    Subscribes to asset (token) IDs and receives real-time price updates.
    """

    def __init__(
        self,
        on_price_update: Optional[Callable[[str, float, float], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the WebSocket monitor.

        Args:
            on_price_update: Callback(asset_id, best_bid, best_ask) for price changes
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.on_price_update = on_price_update

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.ping_thread: Optional[threading.Thread] = None

        self.subscribed_assets: Set[str] = set()
        self.pending_subscriptions: Set[str] = set()
        self.pending_unsubscriptions: Set[str] = set()

        self.is_connected = False
        self.should_run = False
        self.reconnect_delay = 1  # Start with 1 second, exponential backoff

        # Price cache: asset_id -> {bid, ask, mid, last_update}
        self.prices: Dict[str, Dict] = {}

        # Token ID to position mapping for quick lookups
        self.token_to_position: Dict[str, str] = {}

    def subscribe(self, asset_ids: List[str]):
        """
        Subscribe to price updates for given asset IDs.

        Args:
            asset_ids: List of token IDs to subscribe to
        """
        new_assets = set(asset_ids) - self.subscribed_assets
        if not new_assets:
            return

        if self.is_connected and self.ws:
            # Send subscription message immediately
            self._send_subscription(list(new_assets), "subscribe")
            self.subscribed_assets.update(new_assets)
        else:
            # Queue for when connected
            self.pending_subscriptions.update(new_assets)

    def unsubscribe(self, asset_ids: List[str]):
        """
        Unsubscribe from price updates for given asset IDs.

        Args:
            asset_ids: List of token IDs to unsubscribe from
        """
        to_remove = set(asset_ids) & self.subscribed_assets
        if not to_remove:
            return

        if self.is_connected and self.ws:
            self._send_subscription(list(to_remove), "unsubscribe")
            self.subscribed_assets -= to_remove
        else:
            self.pending_unsubscriptions.update(to_remove)

    def _send_subscription(self, asset_ids: List[str], operation: str = "subscribe"):
        """Send subscription/unsubscription message."""
        if not self.ws or not asset_ids:
            return

        msg = {
            "assets_ids": asset_ids,
            "type": "market"
        }

        # For updates after initial connection, include operation
        if operation == "unsubscribe":
            msg["operation"] = "unsubscribe"

        try:
            self.ws.send(json.dumps(msg))
            self.logger.info(f"WebSocket {operation}: {len(asset_ids)} assets")
        except Exception as e:
            self.logger.error(f"Failed to send {operation}: {e}")

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if websocket is None:
            self.logger.error("websocket-client not installed")
            return False

        if self.ws_thread and self.ws_thread.is_alive():
            self.logger.warning("WebSocket already running")
            return True

        self.should_run = True
        self.ws_thread = threading.Thread(target=self._run, daemon=True)
        self.ws_thread.start()
        return True

    def stop(self):
        """Stop the WebSocket connection."""
        self.should_run = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.is_connected = False

    def _run(self):
        """Main WebSocket run loop with reconnection."""
        while self.should_run:
            # Don't connect if there's nothing to subscribe to
            if not self.subscribed_assets and not self.pending_subscriptions:
                self.logger.debug("No assets to subscribe to, waiting...")
                time.sleep(30)
                continue

            try:
                self.ws = websocket.WebSocketApp(
                    CLOB_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                # Use built-in ping (30s interval) for connection keep-alive
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")

            if self.should_run:
                self.logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.is_connected = True

        # Subscribe to all pending + previously subscribed assets
        all_assets = list(self.subscribed_assets | self.pending_subscriptions)
        if all_assets:
            self._send_subscription(all_assets)
            self.subscribed_assets.update(self.pending_subscriptions)
            self.pending_subscriptions.clear()
            self.reconnect_delay = 1  # Only reset backoff after successful subscription
            self.logger.info("WebSocket connected")
        else:
            # No subscriptions - close connection to avoid idle disconnect spam
            self.logger.debug("WebSocket connected but no subscriptions, closing")
            ws.close()
            return

        # Process pending unsubscriptions
        if self.pending_unsubscriptions:
            self._send_subscription(list(self.pending_unsubscriptions), "unsubscribe")
            self.subscribed_assets -= self.pending_unsubscriptions
            self.pending_unsubscriptions.clear()

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Messages can be a list of events or a single event
            events = data if isinstance(data, list) else [data]

            for event in events:
                if not isinstance(event, dict):
                    continue

                event_type = event.get("event_type")

                if event_type == "price_change":
                    self._handle_price_change(event)
                elif event_type == "last_trade_price":
                    self._handle_last_trade(event)
                elif event_type == "book":
                    self._handle_book_snapshot(event)
                # Ignore other event types silently

        except json.JSONDecodeError:
            # Might be a PONG response
            if message != "PONG":
                self.logger.debug(f"Non-JSON message: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _handle_price_change(self, data):
        """Handle price_change event."""
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id")
            if not asset_id:
                continue

            best_bid = float(change.get("best_bid", 0))
            best_ask = float(change.get("best_ask", 0))

            # Calculate midpoint
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask

            # Update cache
            self.prices[asset_id] = {
                "bid": best_bid,
                "ask": best_ask,
                "mid": mid,
                "last_update": datetime.now(timezone.utc).isoformat()
            }

            # Trigger callback
            if self.on_price_update:
                try:
                    self.on_price_update(asset_id, best_bid, best_ask)
                except Exception as e:
                    self.logger.error(f"Price update callback error: {e}")

    def _handle_last_trade(self, data):
        """Handle last_trade_price event."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        price = float(data.get("price", 0))

        # Update cache with last trade
        if asset_id not in self.prices:
            self.prices[asset_id] = {}

        self.prices[asset_id]["last_trade"] = price
        self.prices[asset_id]["last_update"] = datetime.now(timezone.utc).isoformat()

    def _handle_book_snapshot(self, data):
        """Handle book snapshot event."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        # Extract best bid/ask from order book
        # Bids are sorted highest first, asks are sorted lowest first
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Get best bid (highest) and best ask (lowest)
        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[-1].get("price", 0)) if asks else 0  # Last ask is lowest

        # Also get last trade price if available
        last_trade = data.get("last_trade_price")
        if last_trade:
            last_trade = float(last_trade)

        # Calculate mid - prefer using last trade if bid/ask spread is wide
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
        elif last_trade:
            mid = last_trade
        else:
            mid = best_bid or best_ask

        self.prices[asset_id] = {
            "bid": best_bid,
            "ask": best_ask,
            "mid": mid,
            "last_trade": last_trade,
            "last_update": datetime.now(timezone.utc).isoformat()
        }

        # Trigger callback with book data
        if self.on_price_update and (best_bid or best_ask):
            try:
                self.on_price_update(asset_id, best_bid, best_ask)
            except Exception as e:
                self.logger.error(f"Price update callback error: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False

    def _start_ping_thread(self):
        """Start the heartbeat ping thread."""
        if self.ping_thread and self.ping_thread.is_alive():
            return

        def ping_loop():
            while self.should_run and self.is_connected:
                time.sleep(PING_INTERVAL)
                if self.ws and self.is_connected:
                    try:
                        self.ws.send("PING")
                    except:
                        break

        self.ping_thread = threading.Thread(target=ping_loop, daemon=True)
        self.ping_thread.start()

    def get_price(self, asset_id: str) -> Optional[Dict]:
        """
        Get cached price for an asset.

        Returns:
            Dict with bid, ask, mid, last_update or None
        """
        return self.prices.get(asset_id)

    def get_mid_price(self, asset_id: str) -> Optional[float]:
        """Get midpoint price for an asset."""
        price_data = self.prices.get(asset_id)
        return price_data.get("mid") if price_data else None


class PositionPriceTracker:
    """
    Tracks prices for open positions using WebSocket.

    Integrates with the paper trading system to update position prices in real-time.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.monitor = WebSocketPriceMonitor(
            on_price_update=self._on_price_update,
            logger=self.logger
        )

        # Maps token_id -> (position_id, side)
        # side is 'home' or 'away' indicating which team this token represents
        self.token_map: Dict[str, tuple] = {}

        # Position data cache
        self.positions: Dict = {}

    def start(self):
        """Start tracking prices for open positions."""
        self._load_positions()
        self._subscribe_to_positions()
        self.monitor.start()
        self.logger.info("Position price tracker started")

    def stop(self):
        """Stop the price tracker."""
        self.monitor.stop()

    def _load_positions(self):
        """Load positions from file."""
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE, 'r') as f:
                self.positions = json.load(f)

    def _save_positions(self):
        """Save positions to file."""
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)

    def _subscribe_to_positions(self):
        """Subscribe to token IDs for all open positions."""
        token_ids = []

        for pos_id, pos in self.positions.items():
            if pos.get('status') != 'open':
                continue

            bet_side = pos.get('bet_side')
            if not bet_side:
                continue

            # Get the token ID for the side we bet on
            if bet_side == 'home':
                token_id = pos.get('home_token_id')
            else:
                token_id = pos.get('away_token_id')

            if token_id:
                token_ids.append(token_id)
                self.token_map[token_id] = (pos_id, bet_side)

        if token_ids:
            self.monitor.subscribe(token_ids)
            self.logger.info(f"Subscribed to {len(token_ids)} position tokens")

    def _on_price_update(self, asset_id: str, best_bid: float, best_ask: float):
        """Handle price update from WebSocket."""
        if asset_id not in self.token_map:
            return

        pos_id, bet_side = self.token_map[asset_id]

        if pos_id not in self.positions:
            return

        pos = self.positions[pos_id]

        # Calculate current price (use mid)
        current_price = (best_bid + best_ask) / 2

        # Get entry price
        if bet_side == 'home':
            entry_price = pos.get('entry_home_prob', 0)
        else:
            entry_price = pos.get('entry_away_prob', 0)

        if entry_price <= 0:
            return

        # Calculate price change
        price_change = (current_price - entry_price) / entry_price

        # Update position
        pos['current_price_change'] = price_change
        pos['current_price'] = current_price
        pos['last_ws_update'] = datetime.now(timezone.utc).isoformat()

        # Log price for historical analysis
        try:
            from src.Polymarket.price_logger import get_price_logger
            home_team = pos.get('home_team', '')
            away_team = pos.get('away_team', '')
            game_date = pos.get('game_time', '')[:10]
            if bet_side == 'home':
                get_price_logger().log(game_date, home_team, away_team, current_price, 1 - current_price, source="ws")
            else:
                get_price_logger().log(game_date, home_team, away_team, 1 - current_price, current_price, source="ws")
        except Exception:
            pass

        # Track max profit/drawdown
        if 'max_profit_pct' not in pos:
            pos['max_profit_pct'] = price_change
            pos['max_drawdown_pct'] = price_change
        else:
            pos['max_profit_pct'] = max(pos['max_profit_pct'], price_change)
            pos['max_drawdown_pct'] = min(pos['max_drawdown_pct'], price_change)

        # Log significant changes
        if abs(price_change) > 0.1:  # 10%+ change
            home_team = pos.get('home_team', '?')
            away_team = pos.get('away_team', '?')
            self.logger.info(
                f"Price update: {away_team}@{home_team} "
                f"({bet_side}): {entry_price:.1%} -> {current_price:.1%} "
                f"({price_change:+.1%})"
            )

        # Save periodically (debounce in production)
        self._save_positions()

    def refresh_subscriptions(self):
        """Refresh subscriptions for any new positions."""
        self._load_positions()
        self._subscribe_to_positions()


def main():
    """Test the WebSocket monitor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Example: Monitor a specific token
    # You would get this from a position's home_token_id or away_token_id

    def on_update(asset_id, bid, ask):
        mid = (bid + ask) / 2
        print(f"Price update: {asset_id[:20]}... -> bid={bid:.3f}, ask={ask:.3f}, mid={mid:.3f}")

    monitor = WebSocketPriceMonitor(on_price_update=on_update, logger=logger)

    # Subscribe to test token (replace with real token ID)
    # monitor.subscribe(["your_token_id_here"])

    print("Starting WebSocket monitor...")
    print("Press Ctrl+C to stop")

    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()


if __name__ == "__main__":
    main()
