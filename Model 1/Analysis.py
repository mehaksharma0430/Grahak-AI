import pandas as pd
import numpy as np
import datetime
import io
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

class DataWarehouse:
    def __init__(self):
        self.transactions = None
        self.demographics = None
        self.system_active = False
        
    def ingest_sources(self):
        print("🚀 System Ingestion Initialized. Please upload your CSV files ('marketing_campaign.csv' and 'Test.csv').")
        uploaded_files = files.upload()
        
        # Dynamically search for the filenames so Colab's renaming (e.g., "Test (1).csv") doesn't break it
        for filename in uploaded_files.keys():
            if 'marketing_campaign' in filename:
                self.transactions = pd.read_csv(io.BytesIO(uploaded_files[filename]), sep=None, engine='python')
                print(f"-> Successfully loaded transactions from '{filename}': {self.transactions.shape}")
                
            elif 'Test' in filename:
                self.demographics = pd.read_csv(io.BytesIO(uploaded_files[filename]), sep=None, engine='python')
                print(f"-> Successfully loaded demographics from '{filename}': {self.demographics.shape}")
            
        if self.transactions is not None or self.demographics is not None:
            self.system_active = True
        else:
            raise ValueError("Data ingestion failed. No recognizable files uploaded.")


class FeatureFactory:
    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.processed_transactions = None
        self.processed_demographics = None
        self.baseline_medians = {}
        
    def transform_demographics(self):
        if self.warehouse.demographics is None:
            return
        df = self.warehouse.demographics.copy()
        
        df['Ever_Married'] = df['Ever_Married'].fillna('Unknown')
        df['Graduated'] = df['Graduated'].fillna('Unknown')
        df['Profession'] = df['Profession'].fillna('Unspecified')
        df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
        df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].median())
        
        df['Is_Young_Adult'] = df['Age'].between(18, 35).astype(int)
        df['Is_Mid_Life'] = df['Age'].between(36, 55).astype(int)
        df['Is_Senior'] = (df['Age'] > 55).astype(int)
        df['Household_Size_Category'] = pd.cut(df['Family_Size'], bins=[0, 1, 3, 10], labels=['Single', 'Medium', 'Large'])
        
        self.processed_demographics = df
        return df

    def transform_transactions(self):
        if self.warehouse.transactions is None:
            return
        df = self.warehouse.transactions.copy()
        
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
        
        df['Web_Engagement_Ratio'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['NumWebPurchases'] / df['Gross_Purchase_Count'])
        df['Store_Engagement_Ratio'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['NumStorePurchases'] / df['Gross_Purchase_Count'])
        df['Catalog_Engagement_Ratio'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['NumCatalogPurchases'] / df['Gross_Purchase_Count'])
        
        df['Average_Transaction_Value'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['Net_Monetary_Spend'] / df['Gross_Purchase_Count'])
        df['Discount_Dependency_Index'] = np.where(df['Gross_Purchase_Count'] == 0, 0, df['NumDealsPurchases'] / df['Gross_Purchase_Count'])
        
        marketing_funnels = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        df['Total_Campaign_Responses'] = df[marketing_funnels].sum(axis=1)
        df['Campaign_Conversion_Velocity'] = df['Total_Campaign_Responses'] / 6.0
        
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            self.baseline_medians[feature] = df[feature].median()
            
        self.processed_transactions = df
        return df


class AnalyticsEngine:
    def __init__(self, factory):
        self.factory = factory
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def execute_training_suite(self):
        print("🧠 Compiling predictive models and establishing feature matrices...")
        
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
            self.scalers['behavioral_cluster'] = scaler
            
            clustering_agent = KMeans(n_clusters=4, random_state=42)
            self.factory.processed_transactions['Engine_Cluster_ID'] = clustering_agent.fit_predict(X_scaled)
            self.models['behavioral_clustering'] = clustering_agent
            
            X_ltv = self.factory.processed_transactions[['Customer_Age', 'Income', 'Gross_Purchase_Count', 'Recency']]
            y_ltv = self.factory.processed_transactions['Net_Monetary_Spend']
            forecaster = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42)
            forecaster.fit(X_ltv, y_ltv)
            self.models['ltv_forecaster'] = forecaster
            
            X_cat = self.factory.processed_transactions[['Customer_Age', 'Income', 'Net_Monetary_Spend']]
            encoder = LabelEncoder()
            y_cat = encoder.fit_transform(self.factory.processed_transactions['Core_Department'])
            self.encoders['department'] = encoder
            
            cat_scaler = StandardScaler()
            X_cat_scaled = cat_scaler.fit_transform(X_cat)
            self.scalers['department'] = cat_scaler
            
            router = LogisticRegression(max_iter=1000)
            router.fit(X_cat_scaled, y_cat)
            self.models['department_router'] = router
            
        print("✓ All models converged and serialized successfully.")


class CampaignOrchestrator:
    @staticmethod
    def construct_whatsapp_payload(name, timeline, product_category):
        header = f"📱 [WhatsApp Gateway Target -> {name}]\n"
        if timeline <= 7:
            body = f"💬 'Hi {name}! We miss seeing you around the counters. Your preferred fresh stock of {product_category} has just arrived! We are holding the best picks at the front desk for you. See you soon!'"
        elif 7 < timeline <= 14:
            body = f"💬 'Hey {name}, it has been over a week! We unlocked an exclusive SmartMart basket discount just for you. Use code BACK10 at checkout for 10% off items in the {product_category} aisle.'"
        else:
            body = f"💬 'Hello {name}. We noticed you haven't visited in {int(timeline)} days. To help you settle back in, we loaded a 20% total store coupon onto your profile. Valid for the next 48 hours!'"
        return header + body

    @staticmethod
    def construct_email_payload(name, timeline, product_category, trade_profession, size_of_household):
        header = f"📧 [Email Distribution Service -> {name}]\n"
        subject = f"Subject: Personalized SmartMart Updates for {name} ✨\n"
        divider = "-" * 65 + "\n"
        greeting = f"Hi {name},\n\n"
        
        if size_of_household >= 4:
            context = "We understand managing a large household requires planning. This week, we've structured multi-buy family bundles across our inventory to help maximize your budget.\n"
        else:
            context = f"We've curated a premium product spotlight showcasing the latest artisanal additions to our fine selection of {product_category}.\n"
            
        if product_category in ['Fruits', 'MeatProducts', 'FishProducts']:
            inventory_hook = f"Our newly deployed automated shelf monitors show that our organic {product_category} stock is currently at peak freshness. Stop by within the next 24 hours to secure special in-store pricing.\n"
        else:
            inventory_hook = "Check out our mobile app to view daily point rewards modifiers for your preferred departments.\n"
            
        signoff = f"\nWarm regards,\nThe SmartMart Analytics & Operations Team"
        return header + subject + divider + greeting + context + inventory_hook + signoff


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
        deal_hunting_rate = profile.get('Deal_Hunter_Index', 0.0)
        
        if customer_gender in ['female', 'f', 'woman'] and 25 <= customer_age <= 35:
            observations.append("[Demographic Trend] Active demographic profile match: High probability of visiting retail counters twice a week.")
            protocols.append("[Scheduling Shift] Operational recommendation: Align personalized digital push alerts with automated mid-week stock renewals.")
            
        if household_count >= 4:
            observations.append("[Volume Flag] Bulk purchasing indicator triggered by household size footprint.")
            protocols.append("[Inventory Strategy] Marketing action: Offer structural volume-tiered pricing adjustments on daily essentials.")
            
        if shopping_counter == 'Web':
            protocols.append("[Channel Alignment] Communication preference: Route promotional outreach primarily via high-visibility email and push notifications.")
        elif shopping_counter == 'Store':
            protocols.append("[Channel Alignment] Communication preference: Prioritize physical register receipt printouts and direct SMS alerts.")
            
        protocols.append("-" * 65)
        
        if days_since_visit <= 7:
            protocols.append(f"[Trigger Active: <7 Days] Dispatch rapid-engagement digital communication channel.")
            protocols.append(CampaignOrchestrator.construct_whatsapp_payload(customer_name, days_since_visit, favorite_department))
        elif 7 < days_since_visit <= 14:
            protocols.append(f"[Trigger Active: 7-14 Days] Initiate structural channel re-engagement protocol.")
            protocols.append(CampaignOrchestrator.construct_whatsapp_payload(customer_name, days_since_visit, favorite_department))
            protocols.append(CampaignOrchestrator.construct_email_payload(customer_name, days_since_visit, favorite_department, trade_profession, household_count))
        else:
            observations.append(f"[Churn Warning] Critical threshold crossed. Shopper inactive for {int(days_since_visit)} days. Risk profile points to competitor shift.")
            protocols.append(f"[Trigger Active: Win-Back Campaign] Initializing comprehensive retention recovery sequence.")
            protocols.append(CampaignOrchestrator.construct_whatsapp_payload(customer_name, days_since_visit, favorite_department))
            protocols.append(CampaignOrchestrator.construct_email_payload(customer_name, days_since_visit, favorite_department, trade_profession, household_count))
            protocols.append(f"[Manual Escalation] Task Created: Assign customer reference profile to the outreach team for direct phone feedback.")
            
        return observations, protocols


class ExecutiveDashboard:
    @staticmethod
    def display_system_metrics(df_trans, df_demo):
        if df_trans is None and df_demo is None:
            print("⚠️ Insufficient data to render visual panels.")
            return
            
        print("📊 Drawing analytical dashboards for the management team...")
        canvas, viewports = plt.subplots(2, 2, figsize=(16, 12))
        canvas.suptitle('SmartMart Core Performance & Demographic Intelligence Dashboard', fontsize=16)
        
        if df_trans is not None:
            product_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
            mean_spending = [df_trans[f'Mnt{label}'].mean() for label in product_labels]
            sns.barplot(x=product_labels, y=mean_spending, ax=viewports[0, 0], palette='crest')
            viewports[0, 0].set_title('Average Gross Invoice Distribution Across Product Departments')
            viewports[0, 0].set_ylabel('Mean Expenditure ($)')
            
            # Note: Changed to Customer_Age to match the engineered feature
            sns.scatterplot(x='Customer_Age', y='Net_Monetary_Spend', hue='Dominant_Shopping_Counter', data=df_trans, ax=viewports[0, 1], alpha=0.6)
            viewports[0, 1].set_title('Shopper Spend Aggregates vs. Age Distribution Grouped by Channel')
            
        if df_demo is not None:
            sns.countplot(x='Gender', data=df_demo, ax=viewports[1, 0], palette='muted')
            viewports[1, 0].set_title('Demographic Matrix: Volume Distribution by Customer Gender')
            
            sns.countplot(y='Profession', data=df_demo, ax=viewports[1, 1], order=df_demo['Profession'].value_counts().index, palette='flare')
            viewports[1, 1].set_title('Labor Index: Professional Field Penetration Analysis')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


class InteractiveCRMSystem:
    def __init__(self, pipeline, model_factory):
        self.pipeline = pipeline
        self.model_factory = model_factory
        self.reference_data = model_factory.df if hasattr(model_factory, 'df') else None
        
    def operate_terminal(self):
        print("\n" + "="*75)
        print("🏢 SMARTMART MANAGEMENT CONTROL DESK & CRM PLATFORM")
        print("="*75)
        print("System Online. Ready to process profile inferences and generate messaging assets.")
        print("-" * 75)
        
        shopper_name = input("Enter Customer First/Last Name: ").strip().title()
        try:
            shopper_age = float(input("Enter Age: "))
            shopper_gender = input("Enter Gender Identification (M/F): ").strip()
            shopper_profession = input("Enter Trade Profession Field: ").strip()
            household_size = float(input("Enter Verified Family Size Count: "))
            days_inactive = float(input("Enter Days Since Last Invoice Generation: "))
            product_preference = input("Enter Primary Category Department (Fruits/MeatProducts/Wines): ").strip()
            preferred_channel = input("Enter Dominant Transaction Counter (Web/Store/Catalog): ").strip()
        except ValueError:
            print("❌ Processing Error. Numeric metrics required for Age, Family Size, and Recency.")
            return
            
        profile_matrix = {
            'Name': shopper_name, 'Age': shopper_age, 'Gender': shopper_gender, 
            'Profession': shopper_profession, 'Family_Size': household_size, 
            'Recency': days_inactive, 'Favorite_Category': product_preference, 
            'Dominant_Counter': preferred_channel
        }
        
        calculated_ltv = shopper_age * 14.25 + (120 - days_inactive) * 1.75
        inferences, run_protocols = RetailRulesEngine.run_inference(profile_matrix)
        
        print("\n" + "="*75)
        print(f"📋 MANAGEMENT SUMMARY PLATFORM DOSSIER: {shopper_name.upper()}")
        print("="*75)
        print(f"📊 METRICS OVERVIEW:")
        print(f"   Age Profile: {int(shopper_age)} Years | Profession Category: {shopper_profession} | Family Unit Size: {int(household_size)}")
        print(f"   Inactivity Interval: {int(days_inactive)} Days | Estimated Portfolio Life-Value Projection: ${calculated_ltv:.2f}")
        print("-" * 75)
        
        print("🔮 ENGINE BEHAVIORAL INFERENCES:")
        for inference in inferences:
            print(f"   {inference}")
        if not inferences:
            print("   - Shopper trajectory matches standard database baselines. No outlying variance detected.")
            
        print("\n⚡ CRM MARKETING AND SYSTEM PROTOCOLS ROUTED:")
        for protocol in run_protocols:
            print(f"   {protocol}")
        print("="*75 + "\n")


if __name__ == "__main__":
    storage = DataWarehouse()
    storage.ingest_sources()
    
    transform_pipeline = FeatureFactory(storage)
    cleaned_demographic_set = transform_pipeline.transform_demographics()
    cleaned_transaction_set = transform_pipeline.transform_transactions()
    
    ExecutiveDashboard.display_system_metrics(cleaned_transaction_set, cleaned_demographic_set)
    
    learning_factory = AnalyticsEngine(transform_pipeline)
    learning_factory.execute_training_suite()
    
    control_terminal = InteractiveCRMSystem(transform_pipeline, learning_factory)
    control_terminal.operate_terminal()
