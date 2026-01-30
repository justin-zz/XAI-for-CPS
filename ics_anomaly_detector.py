import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import joblib

class ICSUnsupervisedAnomalyDetector:
    """
    Unsupervised anomaly detection for ICS network traffic
    Learns normal behavior from attack-free data, detects attacks as anomalies
    """
    
    def __init__(self, protocol='iec104'):
        self.protocol = protocol
        self.scaler = RobustScaler()  # Robust to outliers
        self.detectors = {}
        self.results = {}
        self.feature_names = None
        
    def parse_headers_file(self, headers_file):
        """
        Parse headers file with features on separate lines, possibly with trailing commas
        """
        selected_features = []
        
        if os.path.exists(headers_file):
            with open(headers_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Remove trailing comma if present
                        if line.endswith(','):
                            line = line[:-1]
                        selected_features.append(line)
        
        # Remove non-feature columns
        non_features = ['Label', 'label', 'is_attack', 'attack_type', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
        selected_features = [f for f in selected_features if f not in non_features]
        
        return selected_features
    
    def load_and_preprocess_data(self, data_dir):
        """
        Load attack-free and attack data from CSV files
        """
        print(f"Loading data for {self.protocol}...")
        
        # Find data directories
        processed_dir = os.path.join(data_dir, 'processed')
        protocol_dir = os.path.join(processed_dir, self.protocol)
        
        attack_free_dir = os.path.join(protocol_dir, 'attack-free-data')
        attack_dir = os.path.join(protocol_dir, 'attack-data')
        
        # Load and parse header file if exists
        headers_file = os.path.join(protocol_dir, f'headers_{self.protocol}.txt')
        selected_features = self.parse_headers_file(headers_file)
        
        print(f"Parsed {len(selected_features)} features from headers file")
        if selected_features:
            print(f"First 10 features: {selected_features[:10]}")
        
        # Load attack-free data (normal behavior)
        normal_data = []
        if os.path.exists(attack_free_dir):
            for file in os.listdir(attack_free_dir):
                if file.endswith('.csv') and 'attackfree' in file.lower():
                    filepath = os.path.join(attack_free_dir, file)
                    df = pd.read_csv(filepath)
                    normal_data.append(df)
                    print(f"  Loaded {file}: {len(df)} samples")
        
        if not normal_data:
            raise ValueError(f"No attack-free data found in {attack_free_dir}")
        
        normal_df = pd.concat(normal_data, ignore_index=True)
        normal_df['is_attack'] = 0  # Label as normal
        
        # Load attack data
        attack_data = []
        attack_labels = {}
        if os.path.exists(attack_dir):
            for file in os.listdir(attack_dir):
                if file.endswith('.csv') and 'attack' in file.lower() and 'attackfree' not in file.lower():
                    filepath = os.path.join(attack_dir, file)
                    df = pd.read_csv(filepath)
                    
                    # Extract attack type from filename
                    # Format: capture104-dosattack.csv
                    attack_type = file.split('-')[1].replace('.csv', '')
                    df['attack_type'] = attack_type
                    df['is_attack'] = 1
                    
                    attack_data.append(df)
                    attack_labels[attack_type] = len(df)
                    print(f"  Loaded {file}: {len(df)} samples ({attack_type})")
        
        attack_df = pd.concat(attack_data, ignore_index=True) if attack_data else pd.DataFrame()
        
        # Combine for analysis
        if not attack_df.empty:
            combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
            print(f"\nTotal samples: {len(combined_df)}")
            print(f"  Normal: {len(normal_df)}")
            print(f"  Attacks: {len(attack_df)}")
            for attack, count in attack_labels.items():
                print(f"    {attack}: {count}")
        else:
            combined_df = normal_df
            print(f"\nOnly normal data available: {len(normal_df)} samples")
        
        # Store metadata
        self.metadata = {
            'normal_count': len(normal_df),
            'attack_count': len(attack_df) if not attack_df.empty else 0,
            'attack_types': attack_labels,
            'selected_features_count': len(selected_features)
        }
        
        return combined_df, normal_df, attack_df, selected_features
    
    def prepare_features(self, df, selected_features=None):
        """
        Prepare features for anomaly detection
        """
        print("\nPreparing features...")
        
        # Drop non-numeric columns that shouldn't be features
        columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
        if 'attack_type' in df.columns:
            columns_to_drop.append('attack_type')
        
        # Keep only columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        # If selected features provided, use them
        if selected_features and len(selected_features) > 0:
            # Ensure selected features exist in dataframe
            available_features = [f for f in selected_features if f in df.columns]
            print(f"Found {len(available_features)}/{len(selected_features)} selected features in data")
            
            if len(available_features) == 0:
                print("Warning: No selected features found in data columns!")
                print(f"Data columns: {list(df.columns)[:20]}...")
                print(f"Looking for: {selected_features[:20]}...")
                
                # Try to match ignoring whitespace/case
                df_cols_lower = [str(col).strip().lower() for col in df.columns]
                selected_lower = [str(f).strip().lower() for f in selected_features]
                
                available_features = []
                for i, feat in enumerate(selected_features):
                    if selected_lower[i] in df_cols_lower:
                        idx = df_cols_lower.index(selected_lower[i])
                        available_features.append(df.columns[idx])
                
                print(f"Found {len(available_features)} features after case-insensitive matching")
            
            # Use selected features plus necessary columns
            features_to_use = available_features + ['is_attack']
            df_features = df[features_to_use].copy()
            self.feature_names = available_features
        else:
            # Use all numeric features except the ones to drop
            df_features = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Keep only numeric columns
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
            df_features = df_features[numeric_cols + ['is_attack']]
            self.feature_names = numeric_cols
        
        # Separate features and labels
        X = df_features.drop(columns=['is_attack'], errors='ignore')
        y = df['is_attack'] if 'is_attack' in df.columns else pd.Series([0] * len(df))
        
        # Check for non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Dropping non-numeric columns: {list(non_numeric)}")
            X = X.drop(columns=non_numeric)
            self.feature_names = [f for f in self.feature_names if f not in non_numeric]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median
        nan_counts = {}
        for col in X.columns:
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                nan_counts[col] = nan_count
        
        if nan_counts:
            print(f"Filled NaN values in {len(nan_counts)} columns")
            max_nan_col = max(nan_counts.items(), key=lambda x: x[1])
            print(f"  Most NaNs in '{max_nan_col[0]}': {max_nan_col[1]} values filled")
        
        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        if len(self.feature_names) > 0:
            print(f"First 10 features: {self.feature_names[:10]}")
        
        return X.values, y.values, X.columns.tolist()
    
    def train_unsupervised_detectors(self, X_normal):
        """
        Train multiple unsupervised anomaly detectors on normal data
        """
        print("\nTraining unsupervised detectors...")
        
        if X_normal.shape[1] == 0:
            raise ValueError("No features available for training! Check feature selection.")
        
        # Scale the data
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        print(f"  Scaled data shape: {X_normal_scaled.shape}")
        
        # 1. Isolation Forest
        print("  Training Isolation Forest...")
        contamination = min(0.1, 100 / len(X_normal))  # At least expect some anomalies
        iso_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_normal_scaled)
        self.detectors['isolation_forest'] = iso_forest
        
        # 2. One-Class SVM
        print("  Training One-Class SVM...")
        nu = min(0.1, 50 / len(X_normal))  # Expected outlier fraction
        oc_svm = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale',
            max_iter=1000
        )
        oc_svm.fit(X_normal_scaled)
        self.detectors['one_class_svm'] = oc_svm
        
        print("Detectors trained successfully!")
        return self.detectors
    
    def detect_anomalies(self, X_test):
        """
        Detect anomalies using all trained detectors
        """
        print("\nDetecting anomalies...")
        
        # Scale the test data
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"  Running {name}...")
            
            try:
                # Get anomaly scores (negative = more anomalous)
                if hasattr(detector, 'decision_function'):
                    scores = -detector.decision_function(X_test_scaled)
                else:
                    scores = -detector.score_samples(X_test_scaled)
                
                # Get binary predictions
                if hasattr(detector, 'predict'):
                    predictions = detector.predict(X_test_scaled)
                    # Convert: 1=normal, -1=anomaly → 0=normal, 1=anomaly
                    binary_pred = np.where(predictions == 1, 0, 1)
                else:
                    # Use threshold on scores (top 5% as anomalies)
                    threshold = np.percentile(scores, 95)
                    binary_pred = (scores > threshold).astype(int)
                
                results[name] = {
                    'scores': scores,
                    'predictions': binary_pred,
                    'threshold': np.percentile(scores, 95)
                }
                
                # Print detection stats
                anomaly_count = binary_pred.sum()
                print(f"    Detected {anomaly_count} anomalies ({anomaly_count/len(binary_pred)*100:.2f}%)")
                
            except Exception as e:
                print(f"    Error with {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.results = results
        return results
    
    def evaluate_detection(self, y_true, results=None):
        """
        Evaluate detection performance (if ground truth available)
        """
        if results is None:
            results = self.results
        
        if y_true is None or len(np.unique(y_true)) < 2:
            print("No ground truth labels available for evaluation")
            return {}
        
        print("\nEvaluating detection performance...")
        
        evaluation = {}
        
        for name, result in results.items():
            y_pred = result['predictions']
            scores = result['scores']
            
            # Calculate metrics
            from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall AUC
            from sklearn.metrics import precision_recall_curve
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall_vals, precision_vals)
            
            evaluation[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Detection Rate: {evaluation[name]['detection_rate']:.4f}")
            print(f"  False Positive Rate: {evaluation[name]['false_positive_rate']:.4f}")
        
        return evaluation
    
    def analyze_feature_importance(self, X_test, detector_name='isolation_forest'):
        """
        Analyze which features contribute most to anomaly detection
        """
        if detector_name not in self.detectors:
            print(f"Detector {detector_name} not found")
            return None
        
        detector = self.detectors[detector_name]
        X_test_scaled = self.scaler.transform(X_test)
        
        if detector_name == 'isolation_forest':
            # Compute feature importance for Isolation Forest
            importances = np.zeros(X_test.shape[1])
            
            for tree in detector.estimators_:
                # Get split features
                tree_feature = tree.tree_.feature
                
                # Count splits per feature
                for node in range(tree.tree_.node_count):
                    if tree_feature[node] >= 0:  # Not leaf
                        importances[tree_feature[node]] += 1
            
            # Normalize
            if importances.sum() > 0:
                importances = importances / importances.sum()
        else:
            # Simple permutation importance as fallback
            baseline_scores = -detector.score_samples(X_test_scaled)
            importances = np.zeros(X_test.shape[1])
            
            for i in range(X_test.shape[1]):
                X_permuted = X_test_scaled.copy()
                np.random.shuffle(X_permuted[:, i])
                perm_scores = -detector.score_samples(X_permuted)
                importances[i] = np.mean(baseline_scores) - np.mean(perm_scores)
            
            # Normalize to 0-1 range
            if importances.max() > importances.min():
                importances = (importances - importances.min()) / (importances.max() - importances.min())
        
        # Create feature importance dataframe
        if len(importances) == len(self.feature_names):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Warning: Importance length {len(importances)} != feature count {len(self.feature_names)}")
            return None
    
    def visualize_results(self, X_test, y_true=None, results=None, save_dir='results'):
        """
        Create visualizations of anomaly detection results
        """
        if results is None:
            results = self.results
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in {save_dir}...")
        
        # 1. PCA visualization of normal vs anomalies
        X_test_scaled = self.scaler.transform(X_test)
        
        # Reduce to 2D if we have many features
        n_components = min(2, X_test_scaled.shape[1])
        if n_components > 0:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_test_scaled)
            
            # Use first detector's predictions
            first_detector = list(results.keys())[0]
            y_pred = results[first_detector]['predictions']
            
            plt.figure(figsize=(12, 10))
            
            # Plot normal vs anomalies
            plt.subplot(2, 2, 1)
            normal_mask = y_pred == 0
            anomaly_mask = y_pred == 1
            
            plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                       alpha=0.5, label='Normal', s=10)
            plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                       alpha=0.7, label='Anomaly', s=20, color='red')
            
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'PCA: Normal vs Anomalies ({first_detector})')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Anomaly score distribution
        plt.subplot(2, 2, 2)
        first_detector = list(results.keys())[0]
        scores = results[first_detector]['scores']
        
        if y_true is not None and len(np.unique(y_true)) >= 2:
            normal_mask = y_true == 0
            attack_mask = y_true == 1
            
            plt.hist(scores[normal_mask], bins=50, alpha=0.7, label='True Normal', density=True)
            plt.hist(scores[attack_mask], bins=50, alpha=0.7, label='True Attack', density=True, color='red')
        else:
            # Use predicted labels
            normal_mask = y_pred == 0
            anomaly_mask = y_pred == 1
            
            plt.hist(scores[normal_mask], bins=50, alpha=0.7, label='Pred Normal', density=True)
            plt.hist(scores[anomaly_mask], bins=50, alpha=0.7, label='Pred Anomaly', density=True, color='red')
        
        threshold = results[first_detector].get('threshold', np.percentile(scores, 95))
        plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.3f}')
        
        plt.xlabel('Anomaly Score (higher = more anomalous)')
        plt.ylabel('Density')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        
        # 3. Feature importance (if available)
        importance_df = self.analyze_feature_importance(X_test, first_detector)
        if importance_df is not None and len(importance_df) > 0:
            plt.subplot(2, 2, 3)
            top_features = importance_df.head(min(10, len(importance_df)))
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Most Important Features')
            plt.gca().invert_yaxis()
        
        # 4. ROC curves for all detectors (if ground truth available)
        if y_true is not None and len(np.unique(y_true)) >= 2:
            plt.subplot(2, 2, 4)
            
            for name, result in results.items():
                scores = result['scores']
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'anomaly_detection_summary_{self.protocol}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved!")
    
    def save_model(self, save_path='ics_anomaly_detector.pkl'):
        """Save the trained model and scaler"""
        save_data = {
            'detectors': self.detectors,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'protocol': self.protocol,
            'metadata': self.metadata
        }
        joblib.dump(save_data, save_path)
        print(f"\nModel saved to {save_path}")
    
    def load_model(self, load_path='ics_anomaly_detector.pkl'):
        """Load a trained model"""
        save_data = joblib.load(load_path)
        self.detectors = save_data['detectors']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.protocol = save_data['protocol']
        self.metadata = save_data['metadata']
        print(f"Model loaded from {load_path}")

# ============================================================================
# DEBUGGING AND ALTERNATIVE APPROACH
# ============================================================================

def debug_feature_matching():
    """
    Debug function to understand why features aren't matching
    """
    print("\n" + "="*70)
    print("DEBUGGING FEATURE MATCHING")
    print("="*70)
    
    # Load a sample CSV to see actual columns
    sample_path = 'processed/iec104/attack-free-data/capture104-attackfree.csv'
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path, nrows=1)  # Just read header
        print(f"\nActual columns in CSV ({len(df.columns)} total):")
        for i, col in enumerate(df.columns[:30]):  # Show first 30
            print(f"  {i:3d}. '{col}'")
        
        # Load headers file
        headers_file = 'processed/iec104/headers_iec104.txt'
        if os.path.exists(headers_file):
            with open(headers_file, 'r') as f:
                headers = [line.strip() for line in f.readlines()]
            
            print(f"\nHeaders from file ({len(headers)} total):")
            for i, header in enumerate(headers[:30]):
                print(f"  {i:3d}. '{header}'")
            
            # Check matches
            print("\nChecking matches...")
            df_cols_lower = [str(col).strip().lower() for col in df.columns]
            headers_lower = [str(h).strip().lower() for h in headers]
            
            matches = []
            for i, header in enumerate(headers):
                header_lower = str(header).strip().lower()
                if header_lower in df_cols_lower:
                    idx = df_cols_lower.index(header_lower)
                    matches.append((header, df.columns[idx]))
            
            print(f"\nFound {len(matches)} matches:")
            for hdr, csv_col in matches[:20]:
                print(f"  File: '{hdr}' -> CSV: '{csv_col}'")
            
            if len(matches) < len(headers):
                print(f"\nMissing {len(headers) - len(matches)} headers:")
                missing = [h for h in headers if str(h).strip().lower() not in df_cols_lower]
                for hdr in missing[:20]:
                    print(f"  '{hdr}'")
    
    return True

def alternative_approach_use_all_numeric():
    """
    Alternative: Just use all numeric columns except obvious non-features
    """
    print("\n" + "="*70)
    print("ALTERNATIVE APPROACH: Using all numeric features")
    print("="*70)
    
    # Initialize detector
    detector = ICSUnsupervisedAnomalyDetector(protocol='iec104')
    
    # Load data without selected features
    combined_df, normal_df, attack_df, _ = detector.load_and_preprocess_data('.')
    
    # Prepare features using all numeric columns
    X_combined, y_combined, feature_names = detector.prepare_features(combined_df, selected_features=None)
    
    print(f"\nUsing ALL numeric features: {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    # Separate normal data for training
    X_normal = X_combined[:detector.metadata['normal_count']]
    print(f"\nNormal data for training: {X_normal.shape}")
    
    # Test data
    X_test = X_combined
    y_test = y_combined
    
    # Train detectors
    detectors = detector.train_unsupervised_detectors(X_normal)
    
    # Detect anomalies
    results = detector.detect_anomalies(X_test)
    
    # Evaluate
    if detector.metadata['attack_count'] > 0:
        evaluation = detector.evaluate_detection(y_test, results)
    
    # Feature importance
    importance_df = detector.analyze_feature_importance(X_test, 'isolation_forest')
    if importance_df is not None:
        print(f"\nTop 10 important features:")
        print(importance_df.head(10).to_string())
    
    # Save results
    os.makedirs('results_alternative', exist_ok=True)
    detector.visualize_results(X_test, y_test, results, save_dir='results_alternative')
    
    return detector

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for unsupervised anomaly detection
    """
    
    # Configuration
    DATA_DIR = '.'  # Current directory
    PROTOCOL = 'iec104'
    SAVE_RESULTS = True
    RESULTS_DIR = 'anomaly_detection_results'
    
    print("=" * 70)
    print("ICS UNSUPERVISED ANOMALY DETECTION")
    print(f"Protocol: {PROTOCOL.upper()}")
    print("=" * 70)
    
    # First, debug feature matching
    debug_feature_matching()
    
    print("\n" + "="*70)
    print("STARTING MAIN ANALYSIS")
    print("="*70)
    
    # Initialize detector
    detector = ICSUnsupervisedAnomalyDetector(protocol=PROTOCOL)
    
    try:
        # 1. Load and preprocess data
        combined_df, normal_df, attack_df, selected_features = detector.load_and_preprocess_data(DATA_DIR)
        
        # 2. Prepare features
        X_combined, y_combined, feature_names = detector.prepare_features(combined_df, selected_features)
        
        if X_combined.shape[1] == 0:
            print("\nERROR: No features found! Trying alternative approach...")
            # Try the alternative approach
            detector = alternative_approach_use_all_numeric()
            return
        
        # Separate normal data for training
        X_normal = X_combined[:detector.metadata['normal_count']]
        print(f"\nNormal data for training: {X_normal.shape}")
        
        # Test data (all data)
        X_test = X_combined
        y_test = y_combined
        
        # 3. Train unsupervised detectors on normal data only
        detectors = detector.train_unsupervised_detectors(X_normal)
        
        # 4. Detect anomalies in all data
        results = detector.detect_anomalies(X_test)
        
        # 5. Evaluate if ground truth available
        if detector.metadata['attack_count'] > 0:
            evaluation = detector.evaluate_detection(y_test, results)
            
            # Print summary
            print("\n" + "=" * 70)
            print("DETECTION SUMMARY")
            print("=" * 70)
            for detector_name, metrics in evaluation.items():
                print(f"\n{detector_name.upper()}:")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                print(f"  Detection Rate: {metrics['detection_rate']:.4f}")
                print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        
        # 6. Analyze feature importance
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        for detector_name in detector.detectors.keys():
            importance_df = detector.analyze_feature_importance(X_test, detector_name)
            if importance_df is not None and len(importance_df) > 0:
                print(f"\nTop 10 important features ({detector_name}):")
                print(importance_df.head(10).to_string())
                
                # Save to CSV
                if SAVE_RESULTS:
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    importance_df.to_csv(
                        os.path.join(RESULTS_DIR, f'feature_importance_{detector_name}_{PROTOCOL}.csv'),
                        index=False
                    )
        
        # 7. Create visualizations
        if SAVE_RESULTS:
            detector.visualize_results(X_test, y_test, results, save_dir=RESULTS_DIR)
        
        # 8. Save model
        if SAVE_RESULTS:
            detector.save_model(os.path.join(RESULTS_DIR, f'ics_anomaly_detector_{PROTOCOL}.pkl'))
        
        print("\n" + "=" * 70)
        print("ANOMALY DETECTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError in main analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach
        print("\nTrying alternative approach with all numeric features...")
        try:
            detector = alternative_approach_use_all_numeric()
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")

if __name__ == "__main__":
    main()
