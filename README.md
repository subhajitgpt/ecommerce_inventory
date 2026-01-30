# PromoOps DS - Inventory & Operations Intelligence Platform

A machine learning-powered dashboard for promotional products operations, providing real-time insights across inventory management, delivery predictions, and production capacity planning.

## Overview

PromoOps DS is a comprehensive Flask-based web application designed for promotional products companies to optimize their entire operations workflow. Built with ML-driven forecasting and AI-powered recommendations, it helps operations teams make data-driven decisions across three critical domains.

## Key Features

### 🎯 Four Integrated Dashboards

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

#### 4. **ML Product Recommendations (NEW!)**

##### **Apriori Market Basket Analysis**
- Association rule mining to discover product purchase patterns
- Identifies frequently bought-together product combinations
- Confidence and lift metrics for rule quality assessment
- Collaborative filtering for personalized recommendations
- Smart bundling suggestions based on transaction history
- Top product combinations ranked by frequency and correlation

**Key Metrics:**
- **Support:** How often products appear together
- **Confidence:** Probability of buying product B given product A
- **Lift:** Strength of association (lift > 1 = positive correlation)

##### **Reinforcement Learning Optimization**

**Q-Learning for Inventory Management:**
- Learns optimal reorder policies through trial-and-error
- 4 states: Critical, Low Stock, Medium Stock, High Stock
- 4 actions: Skip Reorder, Order 500/1000/2000 units
- Trained over 500+ episodes to maximize rewards
- Balances stockout prevention vs. holding costs
- Visual training progress showing reward convergence
- Optimal policy table with Q-values and confidence scores

**Multi-Armed Bandit for Recommendations:**
- UCB (Upper Confidence Bound) algorithm
- Dynamically optimizes product recommendation slots
- Balances exploration (trying new products) vs. exploitation (proven winners)
- Tracks conversion rates and confidence bounds
- Adaptive learning from user interactions
- Real-time performance rankings

**Benefits:**
- **Increased Revenue:** Cross-sell and upsell opportunities from association rules
- **Better Inventory Decisions:** RL-optimized reorder policies reduce costs
- **Catalog Navigation:** Help customers find complementary products faster
- **Adaptive Optimization:** System learns and improves over time
- **Data-Driven Bundling:** Create promotions based on actual purchase patterns
- **Reduced Search Time:** Semantic understanding of product relationships

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
- **ML/AI:** Scikit-learn (Gradient Boosting, K-Means), NumPy, Pandas
- **Algorithms:** Apriori (Association Rules), Q-Learning, Multi-Armed Bandit (UCB)
- **AI Chat:** OpenAI API (GPT-4o-mini)
- **Visualization:** Chart.js 4.4.1 (Line, Bar, Doughnut, Area charts)
- **UI:** Custom dark-themed responsive design with gradient accents

## Machine Learning Components

### **Supervised Learning**
1. **Demand Forecasting:** Gradient Boosting Regressor with lag features and rolling averages for SKU-level predictions
2. **Time-Series Prediction:** Advanced feature engineering for delivery date and capacity forecasting

### **Unsupervised Learning**
3. **Pattern Recognition:** K-Means clustering to group similar inventory behaviors and identify product segments
4. **Anomaly Detection:** Automated identification of outliers in stock levels and turnover rates
5. **Market Basket Analysis:** Apriori algorithm discovers association rules from transaction history
6. **Collaborative Filtering:** Co-occurrence matrix for product recommendation generation

### **Reinforcement Learning**
7. **Q-Learning:** Learns optimal inventory reorder policies by maximizing long-term rewards
   - State space: 4 inventory levels (Critical, Low, Medium, High)
   - Action space: 4 reorder quantities (Skip, 500, 1000, 2000 units)
   - Reward function: Balances stockout penalties and holding costs
   - Exploration strategy: Epsilon-greedy with decay

8. **Multi-Armed Bandit:** UCB algorithm for recommendation slot optimization
   - Upper Confidence Bound (UCB) for exploration-exploitation tradeoff
   - Dynamic learning from conversion rates
   - Adaptive product ranking based on performance

### **AI-Powered Analytics**
9. **Natural Language Processing:** OpenAI GPT-4o-mini for conversational insights
10. **Context-Aware Recommendations:** LLM integration with real-time data for actionable advice

## Algorithm Deep Dive

### Apriori Association Rules
**Purpose:** Discover which products are frequently purchased together

**How it works:**
1. Scans transaction data to find frequent itemsets
2. Generates association rules (IF product A THEN product B)
3. Filters rules by minimum support (frequency) and confidence (reliability)
4. Ranks rules by lift (correlation strength)

**Metrics:**
- **Support:** % of transactions containing the itemset
- **Confidence:** P(B|A) - probability of buying B given A was purchased
- **Lift:** Confidence / P(B) - how much more likely B is purchased with A

### Q-Learning Reinforcement Learning
**Purpose:** Learn optimal inventory reorder policy through experience

**How it works:**
1. Agent observes inventory state (Critical, Low, Medium, High)
2. Takes action (Skip, Order 500/1000/2000 units)
3. Receives reward based on avoiding stockouts and minimizing holding costs
4. Updates Q-table: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
5. Repeats for 500+ episodes until convergence

**Key concepts:**
- **Exploration vs Exploitation:** Epsilon-greedy strategy balances trying new actions vs using learned policy
- **Temporal Difference Learning:** Updates value estimates based on immediate rewards and future predictions
- **Optimal Policy:** After training, agent knows best action for each state

### Multi-Armed Bandit (UCB)
**Purpose:** Optimize which products to recommend in limited slots

**How it works:**
1. Each product is an "arm" of the bandit
2. UCB score = average reward + √(2 ln(total pulls) / arm pulls)
3. Algorithm selects arm with highest UCB score
4. Observes reward (conversion/click) and updates statistics
5. Balance shifts from exploration → exploitation over time

**Why UCB:** Upper confidence bound ensures we try under-explored options while favoring proven performers

## Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd hitpromo

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
python promo.py
```

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

- **Reduce Costs:** Minimize stockouts, overstocking, and overtime expenses through ML optimization
- **Improve Efficiency:** Automate decision-making with supervised, unsupervised, and reinforcement learning
- **Enhance Service:** Proactively manage customer expectations with accurate predictions
- **Data-Driven:** Make informed decisions backed by historical patterns and real-time analytics
- **Increase Revenue:** Cross-sell and upsell with Apriori-based product recommendations
- **Adaptive Learning:** System continuously improves through reinforcement learning
- **Scalable:** Handle growing product catalogs and order volumes seamlessly
- **User-Friendly:** Intuitive dashboard design with no technical expertise required
- **Comprehensive ML:** Covers all major ML paradigms (supervised, unsupervised, reinforcement learning)

## Use Cases Solved

### Catalog Overwhelm
- **Problem:** Customers struggle to find the right promotional items quickly
- **Solution:** Apriori algorithm + semantic search reveals "customers who bought X also bought Y"
- **Impact:** Faster purchase decisions, increased average order value

### Inventory Optimization
- **Problem:** Manual reorder decisions lead to stockouts or excess inventory
- **Solution:** Q-Learning RL agent learns optimal reorder policies from historical data
- **Impact:** Reduced costs, improved service levels, automated decision-making

### Product Discovery
- **Problem:** 15,000+ SKUs make it difficult to surface the right products
- **Solution:** Multi-Armed Bandit dynamically tests and promotes high-converting items
- **Impact:** Better product visibility, data-driven merchandising

### Bundle Creation
- **Problem:** Manual bundle creation based on intuition, not data
- **Solution:** Association rule mining identifies natural product affinities
- **Impact:** Promotions backed by actual purchase patterns
