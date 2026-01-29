#!/usr/bin/env python3
"""
PromoOps Inventory Dashboard - Clean Working Version
Replicates inventory360.lovable.app design with HitPromo branding
"""
from flask import Flask, jsonify, render_template_string, request
import random
import os
import base64
import hashlib
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def decrypt_key(encrypted_data, passphrase="default_salt_2024"):
    """Decrypt a key using XOR with SHA256 hash of passphrase"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        key_hash = hashlib.sha256(passphrase.encode()).digest()
        
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        return decrypted.decode('utf-8')
    except Exception as e:
        return ""

# Load OpenAI client
def get_openai_client():
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        
        # Try encrypted key first
        encrypted_key = os.getenv('OPENAI_API_KEY_ENCRYPTED')
        passphrase = os.getenv('ENCRYPTION_PASSPHRASE', 'default_salt_2024')
        
        if encrypted_key:
            api_key = decrypt_key(encrypted_key, passphrase)
            if api_key:
                return OpenAI(api_key=api_key)
        
        # Fall back to plain key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return OpenAI(api_key=api_key)
            
        return None
    except Exception:
        return None

# Sample data generator with real HitPromo products
def generate_inventory_data():
    products = [
        {"sku": "5790", "name": "20 Oz. Himalayan Tumbler", "colors": ["Navy", "Metallic Orange", "Black", "White", "Red"]},
        {"sku": "50035", "name": "40 Oz. Intrepid Stainless Steel Tumbler", "colors": ["Beige", "White", "Black", "Blue"]},
        {"sku": "50425", "name": "32 Oz. Jasper Recycled Stainless Steel Bottle", "colors": ["Black", "Navy", "Green"]},
        {"sku": "50075", "name": "40 Oz. Jackson Intrepid Stainless Steel Tumbler", "colors": ["Black", "Gray", "Blue"]},
        {"sku": "5900", "name": "16 Oz. Big Game Stadium Cup", "colors": ["White", "Red", "Blue", "Green", "Yellow"]},
        {"sku": "50426", "name": "24 Oz. Torrey Recycled Stainless Steel Bottle", "colors": ["Black", "Navy", "Silver"]},
        {"sku": "50424", "name": "24 Oz. Monterey rPET Bottle", "colors": ["Clear", "Blue", "Green"]},
        {"sku": "5753", "name": "20 Oz. Two-Tone Himalayan Tumbler", "colors": ["Navy/White", "Black/Silver"]},
        {"sku": "50240", "name": "24 Oz. Pop Sip Recycled Stainless Steel Bottle", "colors": ["Black", "White", "Blue"]},
        {"sku": "50078", "name": "40 Oz. Quest Recycled Stainless Steel Tumbler", "colors": ["Black", "Navy", "Gray"]},
    ]
    
    rows = []
    for product in products:
        for color in product["colors"]:
            # Generate historical data for ML model
            historical_demand = [random.randint(800, 1500) for _ in range(12)]  # 12 weeks history
            
            # Use ML to predict next period demand
            forecast = predict_demand_ml(historical_demand)
            
            stock = random.randint(100, 12000)
            turnover = round(random.uniform(0.3, 12.4), 1)
            
            # Determine status based on stock/forecast ratio
            ratio = stock / forecast if forecast > 0 else 0
            if ratio < 0.1:
                status = "Critical"
            elif ratio < 0.5:
                status = "Stockout Risk"
            elif ratio > 2.0:
                status = "Dead Stock"
            elif ratio > 1.5:
                status = "Overstocked"
            else:
                status = "Healthy"
            
            rows.append({
                "sku": product["sku"],
                "name": product["name"],
                "color": color,
                "stock": stock,
                "forecast": int(forecast),
                "turnover": turnover,
                "status": status,
                "historical": historical_demand,
                "ratio": ratio
            })
    
    # Apply clustering to find similar patterns
    rows = apply_clustering(rows)
    
    return rows

def predict_demand_ml(historical_data):
    """Use supervised ML (Gradient Boosting) to predict demand"""
    try:
        if len(historical_data) < 4:
            return sum(historical_data) / len(historical_data)
        
        # Prepare training data with features: week, lag1, lag2, rolling_avg
        X = []
        y = []
        for i in range(3, len(historical_data)):
            features = [
                i,  # week number
                historical_data[i-1],  # lag 1
                historical_data[i-2],  # lag 2
                np.mean(historical_data[max(0, i-4):i])  # rolling avg
            ]
            X.append(features)
            y.append(historical_data[i])
        
        if len(X) < 3:
            return sum(historical_data[-3:]) / 3
        
        # Train Gradient Boosting model
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Predict next period
        last_features = [
            len(historical_data),
            historical_data[-1],
            historical_data[-2],
            np.mean(historical_data[-4:])
        ]
        prediction = model.predict([last_features])[0]
        
        return max(100, prediction)  # Ensure positive prediction
    except Exception:
        return sum(historical_data[-3:]) / 3

def apply_clustering(rows):
    """Use unsupervised ML (K-Means) to cluster items by behavior"""
    try:
        if len(rows) < 3:
            for row in rows:
                row['cluster'] = 0
            return rows
        
        # Features for clustering: turnover, stock/forecast ratio, stock level
        features = []
        for row in rows:
            features.append([
                row['turnover'],
                row['ratio'],
                row['stock'] / 1000  # Normalize stock
            ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply K-Means clustering
        n_clusters = min(3, len(rows))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster info to rows
        for i, row in enumerate(rows):
            row['cluster'] = int(clusters[i])
        
        return rows
    except Exception:
        for row in rows:
            row['cluster'] = 0
        return rows

def generate_recommendations(rows):
    """Generate ML-driven recommendations based on clustering and predictions"""
    recs = []
    
    # Sort by urgency
    critical = [r for r in rows if r["status"] == "Critical"]
    stockout = [r for r in rows if r["status"] == "Stockout Risk"]
    overstocked = [r for r in rows if r["status"] == "Overstocked"]
    dead = [r for r in rows if r["status"] == "Dead Stock"]
    
    # Critical items - highest priority
    for item in critical[:2]:
        reorder_qty = max(1000, int(item['forecast'] * 1.5 - item['stock']))
        recs.append({
            "type": "CRITICAL REORDER",
            "text": f"{item['sku']} {item['color']}: ML forecast {item['forecast']:,} units/month. Current stock {item['stock']:,}. Order {reorder_qty:,} units immediately (Cluster: {item['cluster']}).",
            "class": "critical"
        })
    
    # Stockout risk items
    for item in stockout[:1]:
        reorder_qty = max(800, int(item['forecast'] * 1.2 - item['stock']))
        recs.append({
            "type": "URGENT REORDER",
            "text": f"{item['sku']} {item['color']}: ML model predicts {item['forecast']:,} demand. Stock at {int(item['ratio']*100)}% of forecast. Order {reorder_qty:,} units within 48hrs.",
            "class": "urgent"
        })
    
    # Use clustering insights for overstocked items
    if overstocked:
        # Find items in same cluster as overstocked items
        item = overstocked[0]
        cluster_mates = [r for r in rows if r['cluster'] == item['cluster'] and r['sku'] != item['sku']]
        cluster_info = f" (Similar to {cluster_mates[0]['sku']} pattern)" if cluster_mates else ""
        
        recs.append({
            "type": "SKIP NEXT PO",
            "text": f"{item['sku']} {item['color']}: Stock at {int(item['ratio']*100)}% of ML forecast. {item['turnover']}x turnover{cluster_info}. Skip next 2 reorder cycles.",
            "class": "skip"
        })
    
    # Dead stock - ML identifies low turnover patterns
    if dead:
        item = dead[0]
        savings = int(item['stock'] * 0.3 * 5)  # Estimated 30% discount on $5 avg cost
        recs.append({
            "type": "CLEARANCE CANDIDATE",
            "text": f"{item['sku']} {item['color']}: ML clustering identified as slow-moving (Cluster {item['cluster']}). {item['stock']:,} units, ${savings:,} tied up. Test 30% clearance promo.",
            "class": "clearance"
        })
    
    # Add ML-based insight
    if len(rows) >= 3:
        avg_turnover = np.mean([r['turnover'] for r in rows])
        high_performers = [r for r in rows if r['turnover'] > avg_turnover * 1.5]
        if high_performers:
            item = high_performers[0]
            recs.append({
                "type": "HIGH PERFORMER",
                "text": f"{item['sku']} {item['color']}: ML identified top performer with {item['turnover']}x turnover. Consider increasing safety stock by 20%.",
                "class": "success"
            })
    
    if not recs:
        recs.append({
            "type": "ALL GOOD",
            "text": "ML models show all inventory levels within optimal ranges. No urgent actions needed. Continue monitoring.",
            "class": "success"
        })
    
    return recs

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    rows = generate_inventory_data()
    
    # Calculate KPIs
    stockout = len([r for r in rows if r["status"] in ["Critical", "Stockout Risk"]])
    overstocked = len([r for r in rows if r["status"] in ["Overstocked", "Dead Stock"]])
    healthy = len([r for r in rows if r["status"] == "Healthy"])
    avg_turnover = round(sum(r["turnover"] for r in rows) / len(rows), 1)
    
    # Generate trend data - stacked percentages
    trend = [
        {"week": "Jan 1", "healthy": 30, "risk": 25, "critical": 15, "overstocked": 20, "dead": 10},
        {"week": "Jan 8", "healthy": 35, "risk": 22, "critical": 13, "overstocked": 20, "dead": 10},
        {"week": "Jan 15", "healthy": 32, "risk": 28, "critical": 10, "overstocked": 20, "dead": 10},
        {"week": "Jan 22", "healthy": 38, "risk": 20, "critical": 12, "overstocked": 20, "dead": 10},
        {"week": "Jan 29", "healthy": 40, "risk": 18, "critical": 12, "overstocked": 20, "dead": 10},
    ]
    
    recommendations = generate_recommendations(rows)
    
    return jsonify({
        "kpis": {
            "stockout": stockout,
            "overstocked": overstocked,
            "healthy": healthy,
            "turnover": avg_turnover
        },
        "trend": trend,
        "rows": rows,
        "recommendations": recommendations
    })

@app.route('/api/delivery')
def get_delivery_data():
    """Generate delivery ETA prediction data"""
    orders = [
        {"product": "40 Oz. Intrepid Tumbler", "order_id": "ORD-78234", "customer": "Acme Corp", "qty": 500, "promised": "2025-02-45", "predicted": "2025-02-03", "status": "On-Track", "confidence": 92},
        {"product": "32 Oz. Jasper Recycled Bottle", "order_id": "ORD-78235", "customer": "TechStart Inc", "qty": 250, "promised": "2025-02-18", "predicted": "2025-02-14", "status": "At Risk", "confidence": 67},
        {"product": "16 Oz. Big Game Stadium Cup", "order_id": "ORD-78236", "customer": "Global Events LLC", "qty": 5000, "promised": "2025-02-08", "predicted": "2025-02-07", "status": "On-Track", "confidence": 88},
        {"product": "20 Oz. Himalayan Tumbler", "order_id": "ORD-78237", "customer": "SportsFest LLC", "qty": 2000, "promised": "2025-02-12", "predicted": "2025-02-18", "status": "Delayed", "confidence": 85},
        {"product": "24 Oz. Ricky rPET Bottle", "order_id": "ORD-78238", "customer": "MarketPro Solutions", "qty": 3000, "promised": "2025-02-06", "predicted": "2025-02-05", "status": "On-Track", "confidence": 95},
        {"product": "Top Cup by Ball™ 16 Oz.", "order_id": "ORD-78239", "customer": "HealthFirst Medical", "qty": 1000, "promised": "2025-02-15", "predicted": "2025-02-20", "status": "Delayed", "confidence": 78},
    ]
    
    # Calculate KPIs
    on_track = len([o for o in orders if o["status"] == "On-Track"])
    at_risk = len([o for o in orders if o["status"] == "At Risk"])
    delayed = len([o for o in orders if o["status"] == "Delayed"])
    total = len(orders)
    avg_confidence = round(sum(o["confidence"] for o in orders) / total)
    
    # Performance trend data
    performance_trend = [
        {"week": "Jan 1", "ontime": 85, "atrisk": 10, "delayed": 5},
        {"week": "Jan 8", "ontime": 82, "atrisk": 12, "delayed": 6},
        {"week": "Jan 15", "ontime": 88, "atrisk": 8, "delayed": 4},
        {"week": "Jan 22", "ontime": 80, "atrisk": 15, "delayed": 5},
        {"week": "Jan 29", "ontime": 90, "atrisk": 7, "delayed": 3},
    ]
    
    # Confidence by order
    confidence_by_order = [{"order": o["order_id"], "confidence": o["confidence"]} for o in orders[:8]]
    
    # Alerts
    alerts = [
        {
            "type": "proactive",
            "order_id": "ORD-78237",
            "title": "Proactive Alert: ORD-78237",
            "message": "SportsFest LLC - 2,000x 20 Oz. Himalayan Tumbler predicted to arrive Feb 18 (6 days late). Production bottleneck at Digibrite station.",
            "suggestion": "Suggested: Expedite shipping (+$420) or reallocate to SilkScreen Line 2."
        },
        {
            "type": "watch",
            "order_id": "ORD-78235",
            "title": "Watch: ORD-78235",
            "message": "TechStart Inc - 250x 32 Oz. Jasper Bottle showing 67% confidence. Laser engrave queue at 18 jobs.",
            "suggestion": "Suggested: Contact customer proactively if no improvement by EOD."
        }
    ]
    
    return jsonify({
        "kpis": {
            "on_track": on_track,
            "on_track_pct": round((on_track / total) * 100),
            "at_risk": at_risk,
            "at_risk_pct": round((at_risk / total) * 100),
            "delayed": delayed,
            "delayed_pct": round((delayed / total) * 100),
            "avg_confidence": avg_confidence
        },
        "performance_trend": performance_trend,
        "confidence_by_order": confidence_by_order,
        "orders": orders,
        "alerts": alerts
    })

@app.route('/api/production')
def get_production_data():
    """Generate production capacity planning data"""
    # Station utilization data
    stations = [
        {"id": "ST-B1", "name": "Silk Screen Line 1", "type": "Silk Screen", "queue": 12, "avg_time": 12, "utilization": 94},
        {"id": "ST-B2", "name": "Silk Screen Line 2", "type": "Silk Screen", "queue": 8, "avg_time": 30, "utilization": 87},
        {"id": "ST-B3", "name": "Laser Engrave A", "type": "Laser Engrave", "queue": 18, "avg_time": 15, "utilization": 98},
        {"id": "ST-B4", "name": "Laser Engrave B", "type": "Laser Engrave", "queue": 2, "avg_time": 18, "utilization": 45},
        {"id": "ST-B5", "name": "Digibrite Full Color", "type": "Full-Color Digibrite", "queue": 9, "avg_time": 35, "utilization": 82},
        {"id": "ST-B6", "name": "ColorBrite Press", "type": "ColorBrite", "queue": 6, "avg_time": 22, "utilization": 71},
        {"id": "ST-B7", "name": "Pad Print Station", "type": "Pad Print", "queue": 4, "avg_time": 20, "utilization": 62},
        {"id": "ST-B8", "name": "Sublimation Press", "type": "Sublimation", "queue": 7, "avg_time": 32, "utilization": 78},
    ]
    
    # Current jobs in progress
    jobs = [
        {"job_id": "JOB-1401", "order": "ORD-78234", "product": "40 Oz. Intrepid Tumbler", "qty": 500, "station": "Laser Engrave A", "imprint": "Laser Engrave", "progress": 65},
        {"job_id": "JOB-1402", "order": "ORD-78236", "product": "16 Oz. Big Game Stadium Cup", "qty": 5000, "station": "Silk Screen Line 1", "imprint": "Silk Screen", "progress": 42},
        {"job_id": "JOB-1403", "order": "ORD-78238", "product": "24 Oz. Ricky rPET Bottle", "qty": 3000, "station": "ColorBrite Press", "imprint": "ColorBrite", "progress": 88},
        {"job_id": "JOB-1404", "order": "ORD-78237", "product": "20 Oz. Himalayan Tumbler", "qty": 2000, "station": "Digibrite Full Color", "imprint": "Digibrite", "progress": 12},
    ]
    
    # Calculate KPIs
    avg_utilization = round(sum(s["utilization"] for s in stations) / len(stations))
    total_queue = sum(s["queue"] for s in stations)
    bottleneck_stations = len([s for s in stations if s["utilization"] > 90])
    est_overtime = 12.5  # hours
    
    # Job distribution by method
    method_distribution = [
        {"method": "Silk Screen", "count": 20, "color": "#06b6d4"},
        {"method": "Laser Engrave", "count": 20, "color": "#ef4444"},
        {"method": "Digibrite", "count": 12, "color": "#3b82f6"},
        {"method": "ColorBrite", "count": 10, "color": "#10b981"},
        {"method": "Sublimation", "count": 6, "color": "#f59e0b"},
    ]
    
    # Production alerts
    alerts = [
        {
            "type": "critical",
            "station": "Laser Engrave A",
            "title": "Laser Engrave A: Queue exceeds 4hr capacity - 18 orders waiting",
            "badge": "critical"
        },
        {
            "type": "warning",
            "station": "Silk Screen Line 1",
            "title": "Silk Screen Line 1: Approaching overtime threshold (94% utilization)",
            "badge": "warning"
        },
        {
            "type": "info",
            "station": "Laser Engrave B",
            "title": "Laser Engrave B: Underutilized at 45% - consider rebalancing queue",
            "badge": "info"
        },
        {
            "type": "warning",
            "station": "Digibrite Full Color",
            "title": "Digibrite Full Color: Ink supply running low - reorder recommended",
            "badge": "warning"
        }
    ]
    
    return jsonify({
        "kpis": {
            "avg_utilization": avg_utilization,
            "total_queue": total_queue,
            "bottleneck_stations": bottleneck_stations,
            "est_overtime": est_overtime
        },
        "stations": stations,
        "jobs": jobs,
        "method_distribution": method_distribution,
        "alerts": alerts
    })

@app.route('/api/ai', methods=['POST'])
def ai_chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        client = get_openai_client()
        if not client:
            return jsonify({
                "answer": "OpenAI is not configured. Add OPENAI_API_KEY or OPENAI_API_KEY_ENCRYPTED to your .env file.",
                "source": "local"
            })
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an inventory operations assistant for a promotional products company. Provide concise, actionable advice on inventory management, reorder quantities, and stockout prevention."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return jsonify({
            "answer": response.choices[0].message.content,
            "source": "openai"
        })
    except Exception as e:
        return jsonify({
            "answer": f"Error: {str(e)}",
            "source": "error"
        })

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PromoOps DS - Inventory Forecasting</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
            color: #e8eaf0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: rgba(26, 31, 53, 0.8);
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(100, 120, 180, 0.2);
        }
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 20px;
        }
        .brand h1 { font-size: 22px; font-weight: 600; }
        .brand p { font-size: 13px; color: #94a3b8; margin-top: 2px; }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            overflow-x: auto;
            padding: 5px;
        }
        .tab {
            padding: 12px 20px;
            background: rgba(26, 31, 53, 0.6);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 10px;
            cursor: pointer;
            white-space: nowrap;
            font-size: 14px;
            transition: all 0.3s;
        }
        .tab.active {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border-color: #3b82f6;
        }
        
        /* KPI Cards */
        .kpis {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        .kpi-card {
            background: rgba(26, 31, 53, 0.8);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 16px;
            padding: 24px;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
        }
        .kpi-card.danger::before { background: #ef4444; }
        .kpi-card.warning::before { background: #f59e0b; }
        .kpi-card.success::before { background: #10b981; }
        .kpi-card.info::before { background: #3b82f6; }
        
        .kpi-title { font-size: 13px; color: #94a3b8; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-value { font-size: 36px; font-weight: 700; margin-bottom: 8px; }
        .kpi-subtitle { font-size: 12px; color: #94a3b8; }
        .kpi-change {
            font-size: 13px;
            margin-top: 10px;
            padding: 4px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .kpi-change.up { color: #10b981; background: rgba(16, 185, 129, 0.1); }
        .kpi-change.down { color: #ef4444; background: rgba(239, 68, 68, 0.1); }
        
        /* Chart and Table Grid */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }
        @media (max-width: 1024px) {
            .grid { grid-template-columns: 1fr; }
        }
        
        .chart-container {
            position: relative;
            height: 280px;
            width: 100%;
        }
        
        .panel {
            background: rgba(26, 31, 53, 0.8);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 16px;
            padding: 24px;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .panel-title { font-size: 18px; font-weight: 600; }
        
        .export-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .export-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
        
        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        thead th {
            text-align: left;
            padding: 12px;
            color: #94a3b8;
            font-weight: 600;
            border-bottom: 1px solid rgba(100, 120, 180, 0.2);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        tbody td {
            padding: 14px 12px;
            border-bottom: 1px solid rgba(100, 120, 180, 0.1);
        }
        tbody tr:hover { background: rgba(59, 130, 246, 0.05); }
        
        .product-cell {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .product-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            flex-shrink: 0;
        }
        
        .status-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }
        .status-critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
        .status-risk { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .status-healthy { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
        .status-overstocked { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
        
        /* Recommendations */
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }
        .rec-card {
            background: rgba(26, 31, 53, 0.6);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid;
        }
        .rec-card.critical { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
        .rec-card.urgent { border-left-color: #f97316; background: rgba(249, 115, 22, 0.08); }
        .rec-card.skip { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.05); }
        .rec-card.clearance { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.05); }
        .rec-card.success { border-left-color: #10b981; background: rgba(16, 185, 129, 0.05); }
        
        .rec-type {
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            color: #fff;
        }
        .rec-text {
            font-size: 15px;
            line-height: 1.7;
            color: #e2e8f0;
            font-weight: 500;
        }
        
        /* AI Panel - Floating Assistant */
        .ai-fab {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border-radius: 50%;
            border: 3px solid rgba(59, 130, 246, 0.2);
            color: white;
            font-size: 28px;
            cursor: pointer;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5), 0 0 0 0 rgba(59, 130, 246, 0.4);
            transition: all 0.3s;
            z-index: 9999;
            animation: pulse 2s infinite;
        }
        .ai-fab:hover { 
            transform: scale(1.15);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.6);
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5), 0 0 0 0 rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5), 0 0 0 10px rgba(59, 130, 246, 0); }
        }
        
        .ai-panel {
            position: fixed;
            bottom: 94px;
            right: 24px;
            width: 420px;
            max-width: calc(100vw - 48px);
            background: rgba(26, 31, 53, 0.98);
            border: 1px solid rgba(100, 120, 180, 0.4);
            border-radius: 16px;
            box-shadow: 0 20px 80px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05);
            display: none;
            z-index: 9998;
            backdrop-filter: blur(16px);
            animation: slideUp 0.3s ease-out;
        }
        .ai-panel.visible { display: block; }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .ai-header {
            padding: 16px 20px;
            border-bottom: 1px solid rgba(100, 120, 180, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .ai-header h3 { font-size: 16px; font-weight: 600; }
        .ai-close {
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 28px;
            height: 28px;
        }
        
        .ai-body { padding: 16px 20px; }
        .ai-messages {
            height: 300px;
            overflow-y: auto;
            margin-bottom: 16px;
            padding: 12px;
            background: rgba(10, 14, 26, 0.6);
            border-radius: 12px;
            border: 1px solid rgba(100, 120, 180, 0.1);
        }
        .ai-messages::-webkit-scrollbar { width: 6px; }
        .ai-messages::-webkit-scrollbar-track { background: rgba(26, 31, 53, 0.4); }
        .ai-messages::-webkit-scrollbar-thumb { background: rgba(100, 120, 180, 0.4); border-radius: 3px; }
        
        .ai-message {
            margin-bottom: 16px;
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 14px;
            line-height: 1.6;
        }
        .ai-message.user {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            margin-left: 40px;
            text-align: right;
        }
        .ai-message.assistant {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            margin-right: 40px;
        }
        
        .ai-input-group {
            display: flex;
            gap: 8px;
        }
        .ai-input {
            flex: 1;
            padding: 10px 14px;
            background: rgba(10, 14, 26, 0.6);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 8px;
            color: #e8eaf0;
            font-size: 14px;
        }
        .ai-input:focus {
            outline: none;
            border-color: #3b82f6;
        }
        .ai-send {
            padding: 10px 20px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .ai-send:hover { transform: translateY(-2px); }
        
        .context-btn {
            width: 100%;
            padding: 10px 16px;
            margin-bottom: 12px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            color: #93c5fd;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .context-btn:hover {
            background: rgba(59, 130, 246, 0.2);
            border-color: #3b82f6;
            transform: translateY(-1px);
        }
        
        .table-wrapper { max-height: 350px; overflow-y: auto; }
        .table-wrapper::-webkit-scrollbar { width: 8px; }
        .table-wrapper::-webkit-scrollbar-track { background: rgba(26, 31, 53, 0.4); border-radius: 4px; }
        .table-wrapper::-webkit-scrollbar-thumb { background: rgba(100, 120, 180, 0.4); border-radius: 4px; }
        
        /* Alerts */
        .alerts { margin-top: 20px; }
        .alert {
            padding: 16px 20px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 4px solid;
            font-size: 14px;
            line-height: 1.6;
        }
        .alert.proactive {
            background: rgba(239, 68, 68, 0.1);
            border-left-color: #ef4444;
            color: #fca5a5;
        }
        .alert.watch {
            background: rgba(245, 158, 11, 0.1);
            border-left-color: #f59e0b;
            color: #fbbf24;
        }
        .alert-title {
            font-weight: 700;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .alert-message { margin-bottom: 6px; color: #e2e8f0; }
        .alert-suggestion { color: #94a3b8; font-style: italic; }
        
        /* Tab content visibility */
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .text-right { text-align: right; }
        
        /* Progress Bar */
        .progress-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .progress-bar {
            flex: 1;
            height: 8px;
            background: rgba(100, 120, 180, 0.2);
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            border-radius: 4px;
            transition: width 0.3s;
        }
        .progress-text {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            min-width: 40px;
        }
        
        /* Utilization Bar */
        .utilization-container {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .utilization-bar {
            flex: 1;
            height: 20px;
            background: rgba(100, 120, 180, 0.2);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        .utilization-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .utilization-fill.critical { background: linear-gradient(90deg, #ef4444, #dc2626); }
        .utilization-fill.warning { background: linear-gradient(90deg, #f59e0b, #d97706); }
        .utilization-fill.success { background: linear-gradient(90deg, #10b981, #059669); }
        .utilization-fill.info { background: linear-gradient(90deg, #3b82f6, #2563eb); }
        .utilization-text {
            font-size: 11px;
            font-weight: 700;
            color: #e8eaf0;
            min-width: 45px;
            text-align: right;
        }
        
        /* Alerts Container */
        .alerts-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .alert-item {
            padding: 16px 20px;
            background: rgba(26, 31, 53, 0.6);
            border: 1px solid rgba(100, 120, 180, 0.2);
            border-radius: 12px;
            border-left: 4px solid;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .alert-item.critical { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
        .alert-item.warning { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.05); }
        .alert-item.info { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.05); }
        .alert-icon {
            font-size: 20px;
            flex-shrink: 0;
        }
        .alert-content {
            flex: 1;
        }
        .alert-title {
            font-size: 14px;
            font-weight: 600;
            color: #e8eaf0;
            line-height: 1.5;
        }
        .alert-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .alert-badge.critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .alert-badge.warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
        .alert-badge.info { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
        
        /* Imprint Badge */
        .imprint-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
            border: 1px solid;
        }
        .imprint-silk,
        .imprint-silkscreen { background: rgba(6, 182, 212, 0.2); color: #67e8f9; border-color: rgba(6, 182, 212, 0.3); }
        .imprint-laser,
        .imprint-laserengrave { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border-color: rgba(239, 68, 68, 0.3); }
        .imprint-digibrite,
        .imprint-fullcolordigibrite { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border-color: rgba(59, 130, 246, 0.3); }
        .imprint-colorbrite { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.3); }
        .imprint-sublimation { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
        .imprint-pad,
        .imprint-padprint { background: rgba(168, 85, 247, 0.2); color: #c4b5fd; border-color: rgba(168, 85, 247, 0.3); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <div class="logo-icon">P</div>
                <div class="brand">
                    <h1>PromoOps DS</h1>
                    <p>Powered by HitPromo.net • Inventory SKU×Color Forecasting</p>
                </div>
            </div>
            <div>
                <div style="font-size: 14px; font-weight: 600;">Operations Team</div>
                <div style="font-size: 12px; color: #94a3b8;">Admin</div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab(0)">📊 Inventory SKU×Color Forecasting</div>
            <div class="tab" onclick="switchTab(1)">🚚 Delivery ETA Order Predictions</div>
            <div class="tab" onclick="switchTab(2)">🏭 ML Production Capacity Planning</div>
        </div>
        
        <!-- TAB 1: Inventory Forecasting -->
        <div class="tab-content" id="tab-inventory" style="display: block;">
        <div class="kpis">
            <div class="kpi-card danger">
                <div class="kpi-title">Stockout Risk Items</div>
                <div class="kpi-value" id="kpi-stockout">-</div>
                <div class="kpi-subtitle">SKU×Color combinations</div>
                <div class="kpi-change up">↑ 12% vs last week</div>
            </div>
            <div class="kpi-card warning">
                <div class="kpi-title">Overstocked Items</div>
                <div class="kpi-value" id="kpi-overstocked">-</div>
                <div class="kpi-subtitle">Exceeding 120% of forecast</div>
                <div class="kpi-change down">↓ 8% vs last week</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-title">Healthy Stock</div>
                <div class="kpi-value" id="kpi-healthy">-</div>
                <div class="kpi-subtitle">Within optimal range</div>
                <div class="kpi-change up">↑ 6% vs last week</div>
            </div>
            <div class="kpi-card info">
                <div class="kpi-title">Avg Turnover Rate</div>
                <div class="kpi-value" id="kpi-turnover">-</div>
                <div class="kpi-subtitle">Annual inventory turns</div>
                <div class="kpi-change down">↓ 2% vs last month</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Inventory Health Trend</div>
                </div>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">SKU×Color Inventory Status</div>
                    <button class="export-btn" onclick="exportCSV()">📥 Export CSV</button>
                </div>
                <div class="table-wrapper">
                    <table id="inventoryTable">
                        <thead>
                            <tr>
                                <th>PRODUCT</th>
                                <th>SKU</th>
                                <th>COLOR</th>
                                <th class="text-right">STOCK</th>
                                <th class="text-right">FORECAST</th>
                                <th class="text-right">TURNOVER</th>
                                <th>STATUS</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">AI Recommendations</div>
            </div>
            <div class="recommendations" id="recommendations"></div>
        </div>
        </div>
        
        <!-- TAB 2: Delivery ETA -->
        <div class="tab-content" id="tab-delivery" style="display: none;">
        <div class="kpis">
            <div class="kpi-card success">
                <div class="kpi-title">On-Track Orders</div>
                <div class="kpi-value" id="kpi-ontrack">-</div>
                <div class="kpi-subtitle" id="kpi-ontrack-sub">of active orders</div>
                <div class="kpi-change up">↑ 15% vs last week</div>
            </div>
            <div class="kpi-card warning">
                <div class="kpi-title">At-Risk Orders</div>
                <div class="kpi-value" id="kpi-atrisk">-</div>
                <div class="kpi-subtitle">May miss promised date</div>
                <div class="kpi-change down">↓ 12% vs last week</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-title">Delayed Orders</div>
                <div class="kpi-value" id="kpi-delayed">-</div>
                <div class="kpi-subtitle">Will miss SLA</div>
                <div class="kpi-change up">↑ 8% vs last week</div>
            </div>
            <div class="kpi-card info">
                <div class="kpi-title">Avg Prediction Confidence</div>
                <div class="kpi-value" id="kpi-confidence">-</div>
                <div class="kpi-subtitle">Model accuracy</div>
                <div class="kpi-change up">↑ 3% vs last month</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Delivery Performance Trend</div>
                </div>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Prediction Confidence by Order</div>
                </div>
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">Order ETA Predictions</div>
            </div>
            <div class="table-wrapper">
                <table id="deliveryTable">
                    <thead>
                        <tr>
                            <th>PRODUCT</th>
                            <th>ORDER ID</th>
                            <th>CUSTOMER</th>
                            <th class="text-right">QTY</th>
                            <th>PROMISED</th>
                            <th>PREDICTED</th>
                            <th>STATUS</th>
                        </tr>
                    </thead>
                    <tbody id="deliveryTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="alerts" id="alerts"></div>
        </div>
    </div>
    
    <!-- TAB 3: Production Capacity Planning -->
    <div class="tab-content" id="tab-production" style="display: none;">
        <div class="kpis">
            <div class="kpi-card info">
                <div class="kpi-title">Avg Utilization</div>
                <div class="kpi-value" id="prod-avg-util">-</div>
                <div class="kpi-subtitle">Across all stations</div>
            </div>
            <div class="kpi-card warning">
                <div class="kpi-title">Total Queue</div>
                <div class="kpi-value" id="prod-total-queue">-</div>
                <div class="kpi-subtitle">Jobs waiting</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-title">Bottleneck Stations</div>
                <div class="kpi-value" id="prod-bottlenecks">-</div>
                <div class="kpi-subtitle">>90% utilization</div>
            </div>
            <div class="kpi-card info">
                <div class="kpi-title">Est. Overtime Hours</div>
                <div class="kpi-value" id="prod-overtime">-</div>
                <div class="kpi-subtitle">This week projection</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Station Utilization</div>
                </div>
                <div class="chart-container" style="height: 320px;">
                    <canvas id="stationChart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Job Distribution by Method</div>
                </div>
                <div class="chart-container" style="height: 320px;">
                    <canvas id="methodChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">Current Jobs In Progress</div>
            </div>
            <div class="table-wrapper">
                <table id="jobsTable">
                    <thead>
                        <tr>
                            <th>JOB ID</th>
                            <th>ORDER</th>
                            <th>PRODUCT</th>
                            <th class="text-right">QTY</th>
                            <th>STATION</th>
                            <th>IMPRINT</th>
                            <th>PROGRESS</th>
                        </tr>
                    </thead>
                    <tbody id="jobsTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">Station Details</div>
            </div>
            <div class="table-wrapper">
                <table id="stationsTable">
                    <thead>
                        <tr>
                            <th>STATION ID</th>
                            <th>NAME</th>
                            <th>TYPE</th>
                            <th class="text-right">QUEUE</th>
                            <th class="text-right">AVG TIME (MIN)</th>
                            <th>UTILIZATION</th>
                        </tr>
                    </thead>
                    <tbody id="stationsTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">Production Alerts</div>
            </div>
            <div class="alerts-container" id="productionAlerts"></div>
        </div>
    </div>
    
    <button class="ai-fab" onclick="toggleAI()">✨</button>
    
    <div class="ai-panel" id="aiPanel">
        <div class="ai-header">
            <h3>AI Assistant</h3>
            <button class="ai-close" onclick="toggleAI()">×</button>
        </div>
        <div class="ai-body">
            <div class="ai-messages" id="aiMessages">
                <div class="ai-message assistant">
                    Hi! I can help you analyze inventory, forecast demand, and suggest reorder strategies. What would you like to know?
                </div>
            </div>
            <button class="context-btn" onclick="insertContext()">
                <span style="margin-right: 6px;">📄</span> Insert Context from Current Tab
            </button>
            <div class="ai-input-group">
                <input type="text" class="ai-input" id="aiInput" placeholder="Ask about inventory..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="ai-send" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        let chartInstance = null;
        let performanceChartInstance = null;
        let confidenceChartInstance = null;
        let dataCache = null;
        let deliveryDataCache = null;
        let currentTab = 0;
        
        function switchTab(tabIndex) {
            currentTab = tabIndex;
            
            // Update tab UI
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach((tab, idx) => {
                if (idx === tabIndex) {
                    tab.classList.add('active');
                } else {
                    tab.classList.remove('active');
                }
            });
            
            // Hide all tab contents
            document.getElementById('tab-inventory').style.display = 'none';
            document.getElementById('tab-delivery').style.display = 'none';
            document.getElementById('tab-production').style.display = 'none';
            
            // Show selected tab and load its data
            if (tabIndex === 0) {
                document.getElementById('tab-inventory').style.display = 'block';
                if (!dataCache) loadData();
            } else if (tabIndex === 1) {
                document.getElementById('tab-delivery').style.display = 'block';
                loadDeliveryData();
            } else if (tabIndex === 2) {
                document.getElementById('tab-production').style.display = 'block';
                loadProductionData();
            }
        }
        
        async function loadDeliveryData() {
            const response = await fetch('/api/delivery');
            deliveryDataCache = await response.json();
            
            // Update KPIs
            document.getElementById('kpi-ontrack').textContent = deliveryDataCache.kpis.on_track;
            document.getElementById('kpi-ontrack-sub').textContent = deliveryDataCache.kpis.on_track_pct + '% of active orders';
            document.getElementById('kpi-atrisk').textContent = deliveryDataCache.kpis.at_risk;
            document.getElementById('kpi-delayed').textContent = deliveryDataCache.kpis.delayed;
            document.getElementById('kpi-confidence').textContent = deliveryDataCache.kpis.avg_confidence + '%';
            
            // Render charts
            renderPerformanceChart(deliveryDataCache.performance_trend);
            renderConfidenceChart(deliveryDataCache.confidence_by_order);
            
            // Render table
            renderDeliveryTable(deliveryDataCache.orders);
            
            // Render alerts
            renderAlerts(deliveryDataCache.alerts);
        }
        
        function renderPerformanceChart(data) {
            const ctx = document.getElementById('performanceChart');
            if (performanceChartInstance) performanceChartInstance.destroy();
            
            performanceChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.week),
                    datasets: [
                        {
                            label: 'On Time %',
                            data: data.map(d => d.ontime),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'At Risk %',
                            data: data.map(d => d.atrisk),
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'Delayed %',
                            data: data.map(d => d.delayed),
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e8eaf0', font: { size: 11 }, padding: 10 },
                            position: 'bottom'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#e8eaf0',
                            bodyColor: '#e8eaf0',
                            borderColor: 'rgba(100, 120, 180, 0.3)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function renderConfidenceChart(data) {
            const ctx = document.getElementById('confidenceChart');
            if (confidenceChartInstance) confidenceChartInstance.destroy();
            
            confidenceChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.map(d => d.order),
                    datasets: [{
                        label: 'Confidence %',
                        data: data.map(d => d.confidence),
                        backgroundColor: '#3b82f6',
                        borderColor: '#2563eb',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#e8eaf0',
                            bodyColor: '#e8eaf0',
                            borderColor: 'rgba(100, 120, 180, 0.3)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#94a3b8', font: { size: 10 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function renderDeliveryTable(orders) {
            const tbody = document.getElementById('deliveryTableBody');
            tbody.innerHTML = orders.map(order => {
                let statusClass = 'healthy';
                if (order.status === 'At Risk') statusClass = 'risk';
                else if (order.status === 'Delayed') statusClass = 'critical';
                
                return `
                <tr>
                    <td>
                        <div class="product-cell">
                            <div class="product-icon">${order.product.charAt(0)}</div>
                            <div>${order.product}</div>
                        </div>
                    </td>
                    <td><span style="color: #3b82f6; font-weight: 600;">${order.order_id}</span></td>
                    <td>${order.customer}</td>
                    <td class="text-right">${order.qty.toLocaleString()}</td>
                    <td>${order.promised}</td>
                    <td>${order.predicted}</td>
                    <td><span class="status-badge status-${statusClass}">${order.status}</span></td>
                </tr>
            `}).join('');
        }
        
        function renderAlerts(alerts) {
            const container = document.getElementById('alerts');
            container.innerHTML = alerts.map(alert => `
                <div class="alert ${alert.type}">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-suggestion">${alert.suggestion}</div>
                </div>
            `).join('');
        }
        
        async function loadData() {
            const response = await fetch('/api/data');
            dataCache = await response.json();
            
            // Update KPIs
            document.getElementById('kpi-stockout').textContent = dataCache.kpis.stockout;
            document.getElementById('kpi-overstocked').textContent = dataCache.kpis.overstocked;
            document.getElementById('kpi-healthy').textContent = dataCache.kpis.healthy;
            document.getElementById('kpi-turnover').textContent = dataCache.kpis.turnover + 'x';
            
            // Render chart
            renderChart(dataCache.trend);
            
            // Render table
            renderTable(dataCache.rows);
            
            // Render recommendations
            renderRecommendations(dataCache.recommendations);
        }
        
        function renderChart(data) {
            const ctx = document.getElementById('trendChart');
            if (chartInstance) chartInstance.destroy();
            
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.week),
                    datasets: [
                        {
                            label: 'Healthy',
                            data: data.map(d => d.healthy),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.3)',
                            fill: true,
                            tension: 0.4,
                            stack: 'stack'
                        },
                        {
                            label: 'Stockout Risk',
                            data: data.map(d => d.risk),
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.3)',
                            fill: true,
                            tension: 0.4,
                            stack: 'stack'
                        },
                        {
                            label: 'Critical',
                            data: data.map(d => d.critical),
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.3)',
                            fill: true,
                            tension: 0.4,
                            stack: 'stack'
                        },
                        {
                            label: 'Overstocked',
                            data: data.map(d => d.overstocked),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.3)',
                            fill: true,
                            tension: 0.4,
                            stack: 'stack'
                        },
                        {
                            label: 'Dead Stock',
                            data: data.map(d => d.dead),
                            borderColor: '#94a3b8',
                            backgroundColor: 'rgba(148, 163, 184, 0.3)',
                            fill: true,
                            tension: 0.4,
                            stack: 'stack'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e8eaf0', font: { size: 11 }, padding: 10 },
                            position: 'bottom'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#e8eaf0',
                            bodyColor: '#e8eaf0',
                            borderColor: 'rgba(100, 120, 180, 0.3)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            stacked: true,
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' }
                        },
                        y: {
                            stacked: true,
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function renderTable(rows) {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = rows.map(row => `
                <tr>
                    <td>
                        <div class="product-cell">
                            <div class="product-icon">${row.name.charAt(0)}</div>
                            <div>${row.name}</div>
                        </div>
                    </td>
                    <td>${row.sku}</td>
                    <td>${row.color}</td>
                    <td class="text-right">${row.stock.toLocaleString()}</td>
                    <td class="text-right">${row.forecast.toLocaleString()}</td>
                    <td class="text-right">${row.turnover}x</td>
                    <td><span class="status-badge status-${row.status.toLowerCase().replace(' ', '')}">${row.status}</span></td>
                </tr>
            `).join('');
        }
        
        function renderRecommendations(recs) {
            const container = document.getElementById('recommendations');
            container.innerHTML = recs.map(rec => `
                <div class="rec-card ${rec.class}">
                    <div class="rec-type">${rec.type}</div>
                    <div class="rec-text">${rec.text}</div>
                </div>
            `).join('');
        }
        
        function exportCSV() {
            if (!dataCache) return;
            
            const headers = ['SKU', 'Name', 'Color', 'Stock', 'Forecast', 'Turnover', 'Status'];
            const csv = [
                headers.join(','),
                ...dataCache.rows.map(row => 
                    [row.sku, `"${row.name}"`, row.color, row.stock, row.forecast, row.turnover, row.status].join(',')
                )
            ].join('\\n');
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `inventory_${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function insertContext() {
            const input = document.getElementById('aiInput');
            let context = '';
            
            if (currentTab === 0) {
                // Inventory tab context
                if (dataCache) {
                    const critical = dataCache.rows.filter(r => r.status === 'Critical');
                    const stockout = dataCache.rows.filter(r => r.status === 'Stockout Risk');
                    const overstocked = dataCache.rows.filter(r => r.status === 'Overstocked');
                    
                    context = `Current Inventory Status:\n`;
                    context += `- Stockout Risk: ${dataCache.kpis.stockout} items\n`;
                    context += `- Overstocked: ${dataCache.kpis.overstocked} items\n`;
                    context += `- Healthy: ${dataCache.kpis.healthy} items\n`;
                    context += `- Avg Turnover: ${dataCache.kpis.turnover}x\n\n`;
                    
                    if (critical.length > 0) {
                        context += `Critical Items:\n`;
                        critical.forEach(item => {
                            context += `  - ${item.sku} (${item.color}): ${item.stock} units, forecast ${item.forecast}\n`;
                        });
                    }
                    
                    context += `\nQuestion: `;
                }
            } else if (currentTab === 1) {
                // Delivery tab context
                if (deliveryDataCache) {
                    const delayed = deliveryDataCache.orders.filter(o => o.status === 'Delayed');
                    const atrisk = deliveryDataCache.orders.filter(o => o.status === 'At Risk');
                    
                    context = `Current Delivery Status:\n`;
                    context += `- On-Track: ${deliveryDataCache.kpis.on_track} orders (${deliveryDataCache.kpis.on_track_pct}%)\n`;
                    context += `- At Risk: ${deliveryDataCache.kpis.at_risk} orders (${deliveryDataCache.kpis.at_risk_pct}%)\n`;
                    context += `- Delayed: ${deliveryDataCache.kpis.delayed} orders (${deliveryDataCache.kpis.delayed_pct}%)\n`;
                    context += `- Avg Confidence: ${deliveryDataCache.kpis.avg_confidence}%\n\n`;
                    
                    if (delayed.length > 0) {
                        context += `Delayed Orders:\n`;
                        delayed.forEach(order => {
                            context += `  - ${order.order_id}: ${order.product} (${order.qty} units) for ${order.customer}\n`;
                        });
                    }
                    
                    context += `\nQuestion: `;
                }
            } else if (currentTab === 2) {
                // Production tab context
                if (window.productionDataCache) {
                    const bottlenecks = window.productionDataCache.stations.filter(s => s.utilization > 90);
                    const underutilized = window.productionDataCache.stations.filter(s => s.utilization < 50);
                    
                    context = `Current Production Status:\n`;
                    context += `- Avg Utilization: ${window.productionDataCache.kpis.avg_utilization}%\n`;
                    context += `- Total Queue: ${window.productionDataCache.kpis.total_queue} jobs\n`;
                    context += `- Bottleneck Stations: ${window.productionDataCache.kpis.bottleneck_stations}\n`;
                    context += `- Est. Overtime: ${window.productionDataCache.kpis.est_overtime} hours\n\n`;
                    
                    if (bottlenecks.length > 0) {
                        context += `Bottleneck Stations:\n`;
                        bottlenecks.forEach(station => {
                            context += `  - ${station.name}: ${station.utilization}% utilization, ${station.queue} jobs in queue\n`;
                        });
                    }
                    
                    context += `\nQuestion: `;
                }
            }
            
            if (context) {
                input.value = context;
                input.focus();
                // Move cursor to end
                input.setSelectionRange(input.value.length, input.value.length);
            }
        }
        
        function toggleAI() {
            const panel = document.getElementById('aiPanel');
            panel.classList.toggle('visible');
        }
        
        async function sendMessage() {
            const input = document.getElementById('aiInput');
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            
            // Add user message
            const messagesDiv = document.getElementById('aiMessages');
            const userMsg = document.createElement('div');
            userMsg.className = 'ai-message user';
            userMsg.textContent = message;
            messagesDiv.appendChild(userMsg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Send to API
            try {
                const response = await fetch('/api/ai', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                
                // Add assistant message
                const assistantMsg = document.createElement('div');
                assistantMsg.className = 'ai-message assistant';
                assistantMsg.textContent = data.answer;
                messagesDiv.appendChild(assistantMsg);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } catch (error) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'ai-message assistant';
                errorMsg.textContent = 'Error connecting to AI. Please check your configuration.';
                messagesDiv.appendChild(errorMsg);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        }
        
        async function loadProductionData() {
            const response = await fetch('/api/production');
            const data = await response.json();
            
            // Store for context access
            window.productionDataCache = data;
            
            // Update KPIs
            document.getElementById('prod-avg-util').textContent = data.kpis.avg_utilization + '%';
            document.getElementById('prod-total-queue').textContent = data.kpis.total_queue;
            document.getElementById('prod-bottlenecks').textContent = data.kpis.bottleneck_stations;
            document.getElementById('prod-overtime').textContent = data.kpis.est_overtime;
            
            // Render station utilization chart
            renderStationChart(data.stations);
            
            // Render method distribution chart
            renderMethodChart(data.method_distribution);
            
            // Render jobs table
            renderJobsTable(data.jobs);
            
            // Render stations table
            renderStationsTable(data.stations);
            
            // Render alerts
            renderProductionAlerts(data.alerts);
        }
        
        function renderStationChart(stations) {
            const ctx = document.getElementById('stationChart');
            if (window.stationChartInstance) window.stationChartInstance.destroy();
            
            window.stationChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: stations.map(s => s.name),
                    datasets: [{
                        label: 'Utilization %',
                        data: stations.map(s => s.utilization),
                        backgroundColor: stations.map(s => {
                            if (s.utilization >= 90) return 'rgba(239, 68, 68, 0.8)';
                            if (s.utilization >= 75) return 'rgba(245, 158, 11, 0.8)';
                            if (s.utilization >= 60) return 'rgba(16, 185, 129, 0.8)';
                            return 'rgba(59, 130, 246, 0.8)';
                        }),
                        borderColor: stations.map(s => {
                            if (s.utilization >= 90) return '#ef4444';
                            if (s.utilization >= 75) return '#f59e0b';
                            if (s.utilization >= 60) return '#10b981';
                            return '#3b82f6';
                        }),
                        borderWidth: 2
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#e8eaf0',
                            bodyColor: '#e8eaf0',
                            borderColor: 'rgba(100, 120, 180, 0.3)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(100, 120, 180, 0.1)' },
                            max: 100
                        },
                        y: {
                            ticks: { color: '#e8eaf0', font: { size: 11 } },
                            grid: { display: false }
                        }
                    }
                }
            });
        }
        
        function renderMethodChart(methods) {
            const ctx = document.getElementById('methodChart');
            if (window.methodChartInstance) window.methodChartInstance.destroy();
            
            window.methodChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: methods.map(m => m.method),
                    datasets: [{
                        data: methods.map(m => m.count),
                        backgroundColor: methods.map(m => m.color),
                        borderColor: 'rgba(26, 31, 53, 0.8)',
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e8eaf0', font: { size: 11 }, padding: 12 },
                            position: 'bottom'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#e8eaf0',
                            bodyColor: '#e8eaf0',
                            borderColor: 'rgba(100, 120, 180, 0.3)',
                            borderWidth: 1
                        }
                    }
                }
            });
        }
        
        function renderJobsTable(jobs) {
            const tbody = document.getElementById('jobsTableBody');
            tbody.innerHTML = jobs.map(job => {
                const imprintClass = job.imprint.toLowerCase().replace(' ', '');
                return `
                    <tr>
                        <td>${job.job_id}</td>
                        <td><a href="#" style="color: #3b82f6;">${job.order}</a></td>
                        <td>${job.product}</td>
                        <td class="text-right">${job.qty.toLocaleString()}</td>
                        <td>${job.station}</td>
                        <td><span class="imprint-badge imprint-${imprintClass}">${job.imprint}</span></td>
                        <td>
                            <div class="progress-container">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${job.progress}%"></div>
                                </div>
                                <div class="progress-text">${job.progress}%</div>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }
        
        function renderStationsTable(stations) {
            const tbody = document.getElementById('stationsTableBody');
            tbody.innerHTML = stations.map(station => {
                let utilClass = 'info';
                if (station.utilization >= 90) utilClass = 'critical';
                else if (station.utilization >= 75) utilClass = 'warning';
                else if (station.utilization >= 60) utilClass = 'success';
                
                return `
                    <tr>
                        <td>${station.id}</td>
                        <td>${station.name}</td>
                        <td><span class="imprint-badge imprint-${station.type.toLowerCase().replace(' ', '').replace('-', '')}">${station.type}</span></td>
                        <td class="text-right">${station.queue}</td>
                        <td class="text-right">${station.avg_time}</td>
                        <td>
                            <div class="utilization-container">
                                <div class="utilization-bar">
                                    <div class="utilization-fill ${utilClass}" style="width: ${station.utilization}%"></div>
                                </div>
                                <div class="utilization-text">${station.utilization}%</div>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }
        
        function renderProductionAlerts(alerts) {
            const container = document.getElementById('productionAlerts');
            container.innerHTML = alerts.map(alert => `
                <div class="alert-item ${alert.badge}">
                    <div class="alert-icon">${alert.badge === 'critical' ? '⚠️' : alert.badge === 'warning' ? '⚡' : 'ℹ️'}</div>
                    <div class="alert-content">
                        <div class="alert-title">${alert.title}</div>
                    </div>
                    <div class="alert-badge ${alert.badge}">${alert.badge}</div>
                </div>
            `).join('');
        }
        
        // Load data on page load
        loadData();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5009, debug=True, use_reloader=False)
