# PromoOps DS - Inventory & Operations Intelligence Platform

A machine learning-powered dashboard for promotional products operations, providing real-time insights across inventory management, delivery predictions, and production capacity planning.

## Overview

PromoOps DS is a comprehensive Flask-based web application designed for promotional products companies to optimize their entire operations workflow. Built with ML-driven forecasting and AI-powered recommendations, it helps operations teams make data-driven decisions across three critical domains.

## Key Features

### 🎯 Three Integrated Dashboards

#### 1. **Inventory SKU×Color Forecasting**
- Real-time inventory health monitoring across SKU and color combinations
- ML-driven demand predictions using Gradient Boosting models
- Unsupervised clustering (K-Means) to identify similar product patterns
- Automated reorder recommendations based on stockout risk, turnover rates, and forecast accuracy
- Visual trend analysis showing inventory health over time with stacked area charts
- CSV export functionality for offline analysis

**Benefits:**
- Prevent stockouts before they occur with predictive analytics
- Identify dead stock and overstocked items automatically
- Optimize reorder quantities using historical demand patterns
- Reduce carrying costs while maintaining service levels

#### 2. **Delivery ETA Order Predictions**
- AI-powered delivery date predictions with confidence scoring
- Performance tracking across on-time, at-risk, and delayed orders
- Proactive alert system for potential delays
- Customer-specific order status monitoring
- Trend visualization showing delivery performance over time

**Benefits:**
- Proactively communicate with customers before delays occur
- Identify systemic delivery issues and bottlenecks
- Improve customer satisfaction through transparency
- Reduce expedited shipping costs with better planning

#### 3. **ML Production Capacity Planning**
- Real-time station utilization monitoring across all production methods
- Job queue management and bottleneck identification
- Overtime prediction based on current workload
- Method distribution analysis (Silk Screen, Laser Engrave, Digibrite, etc.)
- Production progress tracking for in-progress jobs
- Automated alerts for overutilized and underutilized stations

**Benefits:**
- Balance workload across production stations
- Reduce overtime costs through better capacity planning
- Identify and resolve bottlenecks in real-time
- Maximize throughput and resource utilization

### 🤖 AI Assistant with Context-Aware Intelligence

- Floating AI chat panel powered by OpenAI GPT-4o-mini
- One-click context insertion from current active tab
- Provides actionable recommendations based on real-time data
- Supports encrypted API key storage for security
- Natural language interface for complex data analysis

**Benefits:**
- Get instant answers about your operations data
- Receive ML-driven insights without manual analysis
- Ask "what if" questions for scenario planning
- Natural language queries eliminate need for technical expertise

### 🔐 Security Features

- XOR + SHA256 encryption for API keys
- Environment variable-based configuration
- No plaintext credential storage
- Secure key management following best practices

## Technology Stack

- **Backend:** Python 3.x, Flask
- **ML/AI:** Scikit-learn (Gradient Boosting, K-Means), NumPy
- **AI Chat:** OpenAI API (GPT-4o-mini)
- **Visualization:** Chart.js 4.4.1
- **UI:** Custom dark-themed responsive design

## Machine Learning Components

1. **Demand Forecasting:** Gradient Boosting Regressor with lag features and rolling averages
2. **Pattern Recognition:** K-Means clustering to group similar inventory behaviors
3. **Anomaly Detection:** Automated identification of outliers in stock levels and turnover rates
4. **Predictive Analytics:** Time-series forecasting for delivery dates and production capacity



Open: `http://127.0.0.1:5009`

## Configuration

Create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
# OR use encrypted key
OPENAI_API_KEY_ENCRYPTED=your_encrypted_key
ENCRYPTION_PASSPHRASE=your_passphrase
```

## Benefits Summary

- **Reduce Costs:** Minimize stockouts, overstocking, and overtime expenses
- **Improve Efficiency:** Automate decision-making with ML-driven recommendations
- **Enhance Service:** Proactively manage customer expectations with accurate predictions
- **Data-Driven:** Make informed decisions backed by historical patterns and real-time analytics
- **Scalable:** Handle growing product catalogs and order volumes seamlessly
- **User-Friendly:** Intuitive dashboard design with no technical expertise required
