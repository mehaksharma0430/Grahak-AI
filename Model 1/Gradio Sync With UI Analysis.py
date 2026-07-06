import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. ML & DATA PIPELINE ENGINE
# ==========================================
class FeatureFactory:
    def __init__(self, transactions_df, demographics_df):
        self.raw_transactions = transactions_df
        self.raw_demographics = demographics_df
        self.processed_transactions = None
        self.processed_demographics = None
        self.baseline_medians = {}
        
    def transform_demographics(self):
        if self.raw_demographics is None:
            return None
        df = self.raw_demographics.copy()
        
        df['Ever_Married'] = df['Ever_Married'].fillna('Unknown')
        df['Graduated'] = df['Graduated'].fillna('Unknown')
        df['Profession'] = df['Profession'].fillna('Unspecified')
        df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
        df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].median())
        
        df['Is_Young_Adult'] = df['Age'].between(18, 35).astype(int)
        df['Is_Mid_Life'] = df['Age'].between(36, 55).astype(int)
        df['Is_Senior'] = (df['Age'] > 55).astype(int)
        
        self.processed_demographics = df
        return df

    def transform_transactions(self):
        if self.raw_transactions is None:
            return None
        df = self.raw_transactions.copy()
        
        calendar_year = datetime.datetime.now().year
        df['Customer_Age'] = calendar_year - df['Year_Birth']
        df = df[(df['Customer_Age'] < 100) & (df['Income'] < 600000)].copy()
        df['Income'] = df['Income'].fillna(df['Income'].median())
        
        spending_departments = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df['Net_Monetary_Spend'] = df[spending_departments].sum(axis=1)
        df['Core_Department'] = df[spending_departments].idxmax(axis=1).str.replace('Mnt', '')
        
        interaction_counters = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        df['Gross_Purchase_Count'] = df[interaction_counters].sum(axis=1)
        df['Dominant_Shopping_Counter'] = df[interaction_counters].idxmax(axis=1).str.replace('Num', '').str.replace('Purchases', '')
        
        df['Discount_Dependency_Index'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['NumDealsPurchases'] / df['Gross_Purchase_Count'])
        
        marketing_funnels = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        df['Total_Campaign_Responses'] = df[marketing_funnels].sum(axis=1)
        
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            self.baseline_medians[feature] = df[feature].median()
            
        self.processed_transactions = df
        return df

class AnalyticsEngine:
    def __init__(self, factory):
        self.factory = factory
        self.models = {}
        
    def execute_training_suite(self):
        if self.factory.processed_demographics is not None:
            features = ['Age', 'Family_Size', 'Work_Experience']
            X = self.factory.processed_demographics[features]
            y = self.factory.processed_demographics['Segmentation']
            classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            classifier.fit(X, y)
            self.models['demographic_segmenter'] = classifier
            
        if self.factory.processed_transactions is not None:
            features = ['Net_Monetary_Spend', 'Gross_Purchase_Count', 'Recency', 'Discount_Dependency_Index']
            X = self.factory.processed_transactions[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            clustering_agent = KMeans(n_clusters=4, random_state=42)
            self.factory.processed_transactions['Engine_Cluster_ID'] = clustering_agent.fit_predict(X_scaled)
            self.models['behavioral_clustering'] = clustering_agent
            
            X_ltv = self.factory.processed_transactions[['Customer_Age', 'Income', 'Gross_Purchase_Count', 'Recency']]
            y_ltv = self.factory.processed_transactions['Net_Monetary_Spend']
            forecaster = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42)
            forecaster.fit(X_ltv, y_ltv)
            self.models['ltv_forecaster'] = forecaster


# ==========================================
# 2. BUSINESS LOGIC & COMMUNICATIONS
# ==========================================
class CampaignOrchestrator:
    @staticmethod
    def construct_whatsapp_payload(name, timeline, product_category):
        if timeline <= 7:
            return f"💬 *'Hi {name}! We miss seeing you around the counters. Your preferred fresh stock of {product_category} has just arrived! We are holding the best picks at the front desk for you. See you soon!'*"
        elif 7 < timeline <= 14:
            return f"💬 *'Hey {name}, it has been over a week! We unlocked an exclusive SmartMart basket discount just for you. Use code BACK10 at checkout for 10% off items in the {product_category} aisle.'*"
        else:
            return f"💬 *'Hello {name}. We noticed you haven't visited in {int(timeline)} days. To help you settle back in, we loaded a 20% total store coupon onto your profile. Valid for the next 48 hours!'*"

    @staticmethod
    def construct_email_payload(name, timeline, product_category, trade_profession, size_of_household):
        if size_of_household >= 4:
            context = "We understand managing a large household requires planning. This week, we've structured multi-buy family bundles across our inventory to help maximize your budget."
        else:
            context = f"We've curated a premium product spotlight showcasing the latest artisanal additions to our fine selection of {product_category}."
            
        inventory_hook = "Check out our mobile app to view daily point rewards modifiers for your preferred departments."
        if product_category in ['Fruits', 'MeatProducts', 'FishProducts']:
            inventory_hook = f"Our newly deployed automated shelf monitors show that our organic {product_category} stock is currently at peak freshness. Stop by within the next 24 hours to secure special in-store pricing."
            
        email_markdown = f"""
**Subject:** Personalized SmartMart Updates for {name} ✨
***
Hi {name},

{context}

{inventory_hook}

Warm regards,  
*The SmartMart Analytics & Operations Team*
"""
        return email_markdown

class RetailRulesEngine:
    @staticmethod
    def run_inference(profile):
        observations = []
        protocols = []
        
        customer_name = profile.get('Name', 'Valued Customer')
        customer_age = profile.get('Age', 30)
        customer_gender = profile.get('Gender', 'Unknown').lower()
        days_since_visit = profile.get('Recency', 0)
        household_count = profile.get('Family_Size', 1)
        favorite_department = profile.get('Favorite_Category', 'General')
        trade_profession = profile.get('Profession', 'Unspecified')
        shopping_counter = profile.get('Dominant_Counter', 'Store')
        
        if customer_gender in ['female', 'f', 'woman'] and 25 <= customer_age <= 35:
            observations.append("🎯 **Demographic Trend:** Active demographic profile match. High probability of visiting retail counters twice a week.")
            protocols.append("⚙️ **Operational:** Align personalized digital push alerts with automated mid-week stock renewals.")
            
        if household_count >= 4:
            observations.append("🛒 **Volume Flag:** Bulk purchasing indicator triggered by household size footprint.")
            protocols.append("🏷️ **Pricing Strategy:** Offer structural volume-tiered pricing adjustments on daily essentials.")
            
        if shopping_counter == 'Web':
            protocols.append("💻 **Channel Alignment:** Route promotional outreach primarily via high-visibility email and push notifications.")
        elif shopping_counter == 'Store':
            protocols.append("🏪 **Channel Alignment:** Prioritize physical register receipt printouts and direct SMS alerts.")
            
        if days_since_visit <= 7:
            protocols.append(f"🟢 **Trigger Active [<7 Days]:** Dispatch rapid-engagement digital communication channel.")
        elif 7 < days_since_visit <= 14:
            protocols.append(f"🟡 **Trigger Active [7-14 Days]:** Initiate structural channel re-engagement protocol.")
        else:
            observations.append(f"🚨 **Churn Warning:** Critical threshold crossed. Shopper inactive for {int(days_since_visit)} days. Risk profile points to competitor shift.")
            protocols.append(f"🔴 **Trigger Active [Win-Back]:** Initializing comprehensive retention recovery sequence.")
            protocols.append("📞 **Manual Escalation:** Task Created -> Assign customer reference profile to the outreach team for direct phone feedback.")
            
        return observations, protocols


# ==========================================
# 3. GLOBAL STATE MANAGER
# ==========================================
class GlobalSystemState:
    def __init__(self):
        self.is_ready = False
        self.transactions_df = None
        self.demographics_df = None
        self.analytics_engine = None

sys_state = GlobalSystemState()


# ==========================================
# 4. GRADIO INTERFACE FUNCTIONS
# ==========================================
def initialize_system(file_paths):
    if not file_paths:
        return None, None, None, None, "⚠️ Please upload the CSV files first."
    
    trans_df, demo_df = None, None
    for f in file_paths:
        file_name = f.name.lower()
        if 'marketing' in file_name or 'campaign' in file_name:
            trans_df = pd.read_csv(f.name, sep=None, engine='python')
        elif 'test' in file_name:
            demo_df = pd.read_csv(f.name, sep=None, engine='python')
            
    if trans_df is None and demo_df is None:
        return None, None, None, None, "⚠️ Could not identify 'marketing_campaign.csv' or 'Test.csv' in uploads."
        
    # Process Data
    factory = FeatureFactory(trans_df, demo_df)
    clean_trans = factory.transform_transactions()
    clean_demo = factory.transform_demographics()
    
    # Train Engine
    engine = AnalyticsEngine(factory)
    engine.execute_training_suite()
    
    # Save to global state
    sys_state.transactions_df = clean_trans
    sys_state.demographics_df = clean_demo
    sys_state.analytics_engine = engine
    sys_state.is_ready = True
    
    # Generate Plots
    fig1, fig2, fig3, fig4 = None, None, None, None
    
    if clean_trans is not None:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        prod_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
        means = [clean_trans[f'Mnt{l}'].mean() for l in prod_labels]
        sns.barplot(x=prod_labels, y=means, ax=ax1, palette='crest')
        ax1.set_title('Average Gross Invoice Distribution Across Product Departments')
        ax1.set_ylabel('Mean Expenditure ($)')
        fig1.tight_layout()
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='Customer_Age', y='Net_Monetary_Spend', hue='Dominant_Shopping_Counter', data=clean_trans, ax=ax2, alpha=0.6)
        ax2.set_title('Shopper Spend Aggregates vs. Age Distribution by Channel')
        fig2.tight_layout()
        
    if clean_demo is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Gender', data=clean_demo, ax=ax3, palette='muted')
        ax3.set_title('Volume Distribution by Customer Gender')
        fig3.tight_layout()
        
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.countplot(y='Profession', data=clean_demo, ax=ax4, order=clean_demo['Profession'].value_counts().index, palette='flare')
        ax4.set_title('Professional Field Penetration Analysis')
        fig4.tight_layout()
        
    return fig1, fig2, fig3, fig4, "✅ System Online. Data Ingested, Features Engineered, and ML Models Trained successfully. You may now proceed to the Executive CRM Terminal."

def generate_crm_dossier(name, age, gender, profession, family_size, recency, category, channel):
    if not sys_state.is_ready:
        return "### ⚠️ System Offline\nPlease go back to the 'System Initialization' tab and upload the required datasets before using the CRM terminal."
        
    profile = {
        'Name': name, 'Age': float(age), 'Gender': gender, 
        'Profession': profession, 'Family_Size': float(family_size), 
        'Recency': float(recency), 'Favorite_Category': category, 
        'Dominant_Counter': channel
    }
    
    ltv = float(age) * 14.25 + (120 - float(recency)) * 1.75
    inferences, protocols = RetailRulesEngine.run_inference(profile)
    
    wa_copy = CampaignOrchestrator.construct_whatsapp_payload(name, float(recency), category)
    email_copy = CampaignOrchestrator.construct_email_payload(name, float(recency), category, profession, float(family_size))
    
    # Format the Output Markdown
    inf_text = "\n".join([f"- {i}" for i in inferences]) if inferences else "- Standard database baseline. No outlying variance detected."
    prot_text = "\n".join([f"- {p}" for p in protocols])
    
    dossier = f"""
## 📋 MANAGEMENT SUMMARY PLATFORM DOSSIER: {name.upper()}
***
### 📊 METRICS OVERVIEW
* **Age Profile:** {int(age)} Years
* **Profession Category:** {profession}
* **Family Unit Size:** {int(family_size)}
* **Inactivity Interval:** {int(recency)} Days
* **Estimated Portfolio Life-Value Projection:** **${max(0, ltv):.2f}**

### 🔮 ENGINE BEHAVIORAL INFERENCES
{inf_text}

### ⚡ CRM MARKETING & SYSTEM PROTOCOLS
{prot_text}
***
### 📤 AUTOMATED COMMUNICATION COPY
**📱 WhatsApp Target Payload:**
> {wa_copy}

**📧 Email Distribution Service:**
{email_copy}
"""
    return dossier


# ==========================================
# 5. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🏢 SmartMart Enterprise CRM & Predictive AI Engine")
    gr.Markdown("Welcome to the SmartMart management platform. Initialize the core systems, then use the terminal to interact with customer profiles.")
    
    with gr.Tabs():
        # TAB 1: DASHBOARD
        with gr.TabItem("⚙️ 1. System Initialization & Dashboards"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Data (marketing_campaign.csv & Test.csv)", file_count="multiple")
                    init_btn = gr.Button("Initialize System", variant="primary")
                    status_out = gr.Textbox(label="System Status", interactive=False)
                    
            gr.Markdown("### 📊 Automated Executive Dashboards")
            with gr.Row():
                plot1 = gr.Plot(label="Revenue Distribution")
                plot2 = gr.Plot(label="Spend vs Age")
            with gr.Row():
                plot3 = gr.Plot(label="Gender Demographics")
                plot4 = gr.Plot(label="Profession Matrix")
                
            init_btn.click(
                fn=initialize_system,
                inputs=[file_input],
                outputs=[plot1, plot2, plot3, plot4, status_out]
            )

        # TAB 2: CRM TERMINAL
        with gr.TabItem("💻 2. Executive CRM Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Customer Profile")
                    name_in = gr.Textbox(label="Customer Name", placeholder="e.g., Jane Doe")
                    age_in = gr.Number(label="Age", value=30)
                    gender_in = gr.Dropdown(choices=["Female", "Male", "Other"], label="Gender", value="Female")
                    prof_in = gr.Dropdown(choices=["Healthcare", "Engineer", "Executive", "Marketing", "Doctor", "Lawyer", "Other"], label="Profession", value="Healthcare")
                    fam_in = gr.Number(label="Family Size", value=4)
                    recency_in = gr.Number(label="Days Since Last Visit", value=15)
                    cat_in = gr.Dropdown(choices=["Wines", "Fruits", "MeatProducts", "FishProducts", "SweetProducts"], label="Favorite Category", value="Fruits")
                    channel_in = gr.Dropdown(choices=["Store", "Web", "Catalog"], label="Preferred Channel", value="Store")
                    
                    generate_btn = gr.Button("Run Inference & Generate Dossier", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.Markdown("### Executive Output Dossier")
                    dossier_out = gr.Markdown(value="Waiting for system input...")
                    
            generate_btn.click(
                fn=generate_crm_dossier,
                inputs=[name_in, age_in, gender_in, prof_in, fam_in, recency_in, cat_in, channel_in],
                outputs=[dossier_out]
            )

# Launch the app with a public shareable link
demo.launch(share=True)
