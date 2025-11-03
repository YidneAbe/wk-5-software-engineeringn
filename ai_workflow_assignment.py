"""
AI Development Workflow Assignment
Real-World Problem: Customer Churn Prediction
"""

# =============================================================================
# STAGE 1: PROBLEM DEFINITION & PLANNING
# =============================================================================

class ProblemDefinition:
    """
    Stage 1: Define the business problem, objectives, and success metrics
    """
    
    def __init__(self):
        self.business_problem = "Customer churn in telecommunications industry"
        self.objectives = [
            "Predict which customers are likely to churn",
            "Identify key factors driving customer churn",
            "Provide actionable insights to retention teams"
        ]
        self.success_metrics = {
            "accuracy": "> 85%",
            "precision": "> 80%",
            "recall": "> 75%",
            "business_impact": "Reduce churn by 15%"
        }
    
    def display_problem_statement(self):
        print("=" * 60)
        print("STAGE 1: PROBLEM DEFINITION")
        print("=" * 60)
        print(f"Business Problem: {self.business_problem}")
        print("\nObjectives:")
        for i, obj in enumerate(self.objectives, 1):
            print(f"{i}. {obj}")
        print("\nSuccess Metrics:")
        for metric, target in self.success_metrics.items():
            print(f"- {metric}: {target}")

# =============================================================================
# STAGE 2: DATA COLLECTION & PREPARATION
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Stage 2: Data collection, cleaning, and preparation
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic customer churn data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'monthly_charges': np.random.normal(65, 25, n_samples),
            'total_charges': np.random.normal(2000, 1000, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'monthly_charges_quartile': np.random.choice([1, 2, 3, 4], n_samples)
        }
        
        self.data = pd.DataFrame(data)
        
        # Create target variable based on features
        churn_prob = (
            0.3 * (self.data['contract_type'] == 'Month-to-month') +
            0.2 * (self.data['internet_service'] == 'Fiber optic') +
            0.1 * (self.data['online_security'] == 'No') +
            0.1 * (self.data['tech_support'] == 'No') +
            0.05 * (self.data['monthly_charges_quartile'] == 4) -
            0.1 * (self.data['tenure'] > 24)
        )
        
        self.data['churn'] = (churn_prob + np.random.normal(0, 0.1, n_samples)) > 0.4
        
        return self.data
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n" + "=" * 60)
        print("STAGE 2: DATA PREPROCESSING")
        print("=" * 60)
        
        # Handle categorical variables
        categorical_cols = ['contract_type', 'internet_service', 'online_security', 'tech_support']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le
        
        # Prepare features and target
        X = self.data.drop('churn', axis=1)
        y = self.data['churn']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['age', 'monthly_charges', 'total_charges', 'tenure']
        self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
        
        print("Data preprocessing completed!")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]} columns")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

# =============================================================================
# STAGE 3: MODEL SELECTION & TRAINING
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class ModelTrainer:
    """
    Stage 3: Model selection, training, and hyperparameter tuning
    """
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        self.cv_scores = {}
        
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n" + "=" * 60)
        print("STAGE 3: MODEL TRAINING")
        print("=" * 60)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate using cross-validation
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # Cross-validation scores
            cv_score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            self.cv_scores[name] = cv_score
            
            print(f"{name}:")
            print(f"  CV Accuracy: {cv_score.mean():.3f} (+/- {cv_score.std() * 2:.3f})")
        
        return self.models

# =============================================================================
# STAGE 4: MODEL EVALUATION
# =============================================================================

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Stage 4: Comprehensive model evaluation and interpretation
    """
    
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        
    def evaluate_models(self):
        """Evaluate all trained models on test set"""
        print("\n" + "=" * 60)
        print("STAGE 4: MODEL EVALUATION")
        print("=" * 60)
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name} Performance:")
            print(f"Accuracy: {self.results[name]['accuracy']:.3f}")
            print(f"ROC AUC: {self.results[name]['roc_auc']:.3f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Not Churn', 'Churn']))
        
        return self.results
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()

# =============================================================================
# STAGE 5: DEPLOYMENT CONSIDERATIONS
# =============================================================================

class DeploymentPlan:
    """
    Stage 5: Deployment strategy and monitoring plan
    """
    
    def __init__(self, best_model):
        self.best_model = best_model
        self.deployment_strategy = {
            'environment': 'AWS SageMaker',
            'api_type': 'REST API',
            'monitoring': 'Continuous performance tracking',
            'retraining': 'Monthly model refresh'
        }
    
    def create_deployment_plan(self):
        print("\n" + "=" * 60)
        print("STAGE 5: DEPLOYMENT PLAN")
        print("=" * 60)
        
        print("Deployment Strategy:")
        for key, value in self.deployment_strategy.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
        
        print("\nAPI Endpoint Design:")
        print("POST /predict-churn")
        print("Request: Customer features JSON")
        print("Response: {'churn_probability': float, 'prediction': bool, 'risk_level': str}")
        
        print("\nMonitoring Metrics:")
        monitor_metrics = [
            "API response time < 200ms",
            "Model accuracy drift detection",
            "Feature distribution monitoring",
            "Business impact tracking"
        ]
        for metric in monitor_metrics:
            print(f"- {metric}")

# =============================================================================
# STAGE 6: ETHICAL CONSIDERATIONS & CHALLENGES
# =============================================================================

class EthicalAnalysis:
    """
    Stage 6: Analyze ethical implications and challenges
    """
    
    def __init__(self):
        self.ethical_considerations = [
            "Data privacy: Customer data protection and GDPR compliance",
            "Bias and fairness: Ensure model doesn't discriminate against any group",
            "Transparency: Explainable AI for customer churn predictions",
            "Consent: Proper disclosure of data usage for AI modeling"
        ]
        
        self.challenges = [
            "Imbalanced data: Churn is typically rare (10-20% of customers)",
            "Concept drift: Customer behavior changes over time",
            "Feature engineering: Creating meaningful predictors from raw data",
            "Model interpretability: Balancing accuracy with explainability"
        ]
    
    def present_analysis(self):
        print("\n" + "=" * 60)
        print("STAGE 6: ETHICAL CONSIDERATIONS & CHALLENGES")
        print("=" * 60)
        
        print("\nEthical Considerations:")
        for i, consideration in enumerate(self.ethical_considerations, 1):
            print(f"{i}. {consideration}")
        
        print("\nTechnical Challenges:")
        for i, challenge in enumerate(self.challenges, 1):
            print(f"{i}. {challenge}")
        
        print("\nMitigation Strategies:")
        mitigations = [
            "Regular bias audits and fairness testing",
            "Implement model explainability tools (SHAP, LIME)",
            "Data anonymization and privacy-preserving techniques",
            "Continuous monitoring and model retraining"
        ]
        for mitigation in mitigations:
            print(f"- {mitigation}")

# =============================================================================
# MAIN EXECUTION & WORKFLOW DEMONSTRATION
# =============================================================================

def main():
    """
    Execute the complete AI Development Workflow
    """
    print("AI DEVELOPMENT WORKFLOW: CUSTOMER CHURN PREDICTION")
    print("=" * 60)
    
    # Stage 1: Problem Definition
    problem = ProblemDefinition()
    problem.display_problem_statement()
    
    # Stage 2: Data Processing
    processor = DataProcessor()
    data = processor.generate_sample_data(1000)
    X_train, X_test, y_train, y_test = processor.preprocess_data()
    
    # Stage 3: Model Training
    trainer = ModelTrainer(X_train, y_train)
    models = trainer.train_models()
    
    # Stage 4: Model Evaluation
    evaluator = ModelEvaluator(models, X_test, y_test)
    results = evaluator.evaluate_models()
    
    # Select best model based on ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    
    # Stage 5: Deployment
    deployment = DeploymentPlan(best_model)
    deployment.create_deployment_plan()
    
    # Stage 6: Ethical Analysis
    ethics = EthicalAnalysis()
    ethics.present_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETION SUMMARY")
    print("=" * 60)
    print("✓ Problem defined and success metrics established")
    print("✓ Data collected, cleaned, and preprocessed")
    print("✓ Multiple models trained and validated")
    print("✓ Comprehensive model evaluation completed")
    print("✓ Deployment strategy designed")
    print("✓ Ethical considerations analyzed")
    print("\nThe AI Development Workflow has been successfully demonstrated!")

if __name__ == "__main__":
    main()