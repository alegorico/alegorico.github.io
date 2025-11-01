---
layout: post
title: Scikit-learn - Machine Learning Clásico que Nunca Pasa de Moda
tags: [scikit-learn, machine-learning, data-science, python, algorithms, classification]
---

**Scikit-learn** sigue siendo la biblioteca fundamental para machine learning en Python. Mientras los modelos de deep learning captan la atención, los algoritmos clásicos de scikit-learn resuelven el 80% de los problemas reales de ML de manera eficiente y interpretable.

## Fundamentos Renovados de Scikit-learn

### 1. Configuración y Mejores Prácticas 2024

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, RFECV, 
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, LinearRegression
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE, UMAP

# Métricas y evaluación
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, mean_squared_error,
    mean_absolute_error, r2_score
)

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("Scikit-learn version:", sklearn.__version__)
```

### 2. Pipeline Completo de Machine Learning

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPipeline:
    """Pipeline completo de ML con mejores prácticas"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.pipeline = None
        self.best_model = None
        self.feature_names = None
        self.target_names = None
        self.results = {}
        
    def create_preprocessing_pipeline(self, numeric_features, categorical_features):
        """Crear pipeline de preprocesamiento robusto"""
        
        # Transformadores para variables numéricas
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),  # Más robusto que StandardScaler
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ('selector', VarianceThreshold())  # Eliminar features con baja varianza
        ])
        
        # Transformadores para variables categóricas
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        
        # Combinar transformadores
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        return preprocessor
    
    def get_models_for_comparison(self):
        """Obtener modelos optimizados para comparación"""
        
        if self.problem_type == 'classification':
            models = {
                'LogisticRegression': LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'NaiveBayes': GaussianNB(),
                'AdaBoost': AdaBoostClassifier(random_state=42)
            }
        else:  # regression
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
                'SVR': SVR()
            }
            
        return models
    
    def compare_models(self, X, y, cv_folds=5):
        """Comparar múltiples modelos con validación cruzada"""
        
        models = self.get_models_for_comparison()
        results = []
        
        # Configurar validación cruzada estratificada
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        print("Comparando modelos...")
        for name, model in models.items():
            # Validación cruzada
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            results.append({
                'Model': name,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'CV_Scores': cv_scores
            })
            
            print(f"{name:20} - Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Convertir a DataFrame y ordenar
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('CV_Mean', ascending=False)
        
        self.results['model_comparison'] = results_df
        return results_df
    
    def optimize_hyperparameters(self, X, y, model_name='RandomForest'):
        """Optimizar hiperparámetros del mejor modelo"""
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
        }
        
        models_map = self.get_models_for_comparison()
        
        if model_name not in param_grids:
            print(f"No hay grid de parámetros definido para {model_name}")
            return None
        
        print(f"Optimizando hiperparámetros para {model_name}...")
        
        # Usar RandomizedSearchCV para eficiencia
        model = models_map[model_name]
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grids[model_name],
            n_iter=50,  # Número de combinaciones a probar
            cv=5,
            scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        print(f"Mejores parámetros: {random_search.best_params_}")
        print(f"Mejor score: {random_search.best_score_:.4f}")
        
        self.best_model = random_search.best_estimator_
        self.results['best_params'] = random_search.best_params_
        self.results['best_score'] = random_search.best_score_
        
        return random_search
    
    def feature_importance_analysis(self, X, y, feature_names=None):
        """Análisis de importancia de características"""
        
        if self.best_model is None:
            print("Primero entrena un modelo usando optimize_hyperparameters()")
            return None
        
        # Obtener importancias si el modelo las soporta
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Crear DataFrame de importancias
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Visualizar top 20 características
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia')
            plt.title(f'Top 20 Características - {type(self.best_model).__name__}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            self.results['feature_importance'] = importance_df
            return importance_df
        
        else:
            print("El modelo seleccionado no soporta feature importance")
            return None
    
    def learning_curve_analysis(self, X, y):
        """Análisis de curvas de aprendizaje"""
        
        if self.best_model is None:
            print("Primero entrena un modelo")
            return None
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, n_jobs=-1, random_state=42
        )
        
        # Calcular medias y desviaciones
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Visualizar curvas de aprendizaje
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self.results['learning_curves'] = {
            'train_sizes': train_sizes,
            'train_scores': (train_mean, train_std),
            'val_scores': (val_mean, val_std)
        }
        
        return train_sizes, train_scores, val_scores
    
    def create_ensemble(self, X, y, top_n=3):
        """Crear ensemble con los mejores modelos"""
        
        if 'model_comparison' not in self.results:
            print("Primero ejecuta compare_models()")
            return None
        
        # Obtener top N modelos
        top_models = self.results['model_comparison'].head(top_n)
        models_map = self.get_models_for_comparison()
        
        estimators = []
        for _, row in top_models.iterrows():
            model_name = row['Model']
            model = models_map[model_name]
            estimators.append((model_name.lower(), model))
        
        # Crear ensemble con voting
        if self.problem_type == 'classification':
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Para usar probabilidades
            )
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        # Evaluar ensemble con validación cruzada
        cv_scores = cross_val_score(
            ensemble, X, y, 
            cv=5, 
            scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        )
        
        print(f"Ensemble CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Entrenar ensemble final
        ensemble.fit(X, y)
        self.results['ensemble'] = ensemble
        self.results['ensemble_score'] = cv_scores.mean()
        
        return ensemble

# Ejemplo completo con dataset real
def demo_advanced_pipeline():
    """Demostración completa del pipeline avanzado"""
    
    # Cargar dataset (usando make_classification para demo)
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convertir a DataFrame para trabajar más fácilmente
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print("=== Pipeline Avanzado de Machine Learning ===")
    print(f"Dataset: {df.shape[0]} muestras, {df.shape[1]-1} características")
    print(f"Clases: {np.unique(y)}")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crear pipeline
    pipeline = AdvancedMLPipeline(problem_type='classification')
    
    # 1. Comparar modelos
    print("\n1. Comparación de Modelos:")
    model_comparison = pipeline.compare_models(X_train, y_train)
    print("\nTop 3 modelos:")
    print(model_comparison.head(3)[['Model', 'CV_Mean', 'CV_Std']])
    
    # 2. Optimizar el mejor modelo
    print("\n2. Optimización de Hiperparámetros:")
    best_model_name = model_comparison.iloc[0]['Model']
    optimized_model = pipeline.optimize_hyperparameters(X_train, y_train, best_model_name)
    
    # 3. Análisis de características
    print("\n3. Análisis de Importancia de Características:")
    importance_analysis = pipeline.feature_importance_analysis(X_train, y_train, feature_names)
    if importance_analysis is not None:
        print("Top 5 características más importantes:")
        print(importance_analysis.head())
    
    # 4. Curvas de aprendizaje
    print("\n4. Análisis de Curvas de Aprendizaje:")
    learning_curves = pipeline.learning_curve_analysis(X_train, y_train)
    
    # 5. Crear ensemble
    print("\n5. Creación de Ensemble:")
    ensemble = pipeline.create_ensemble(X_train, y_train, top_n=3)
    
    # 6. Evaluación final en test set
    print("\n6. Evaluación Final:")
    
    # Modelo individual optimizado
    y_pred_individual = pipeline.best_model.predict(X_test)
    individual_accuracy = accuracy_score(y_test, y_pred_individual)
    
    # Ensemble
    if ensemble is not None:
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"Accuracy modelo individual: {individual_accuracy:.4f}")
        print(f"Accuracy ensemble: {ensemble_accuracy:.4f}")
        print(f"Mejora con ensemble: {ensemble_accuracy - individual_accuracy:.4f}")
    else:
        print(f"Accuracy modelo individual: {individual_accuracy:.4f}")
    
    return pipeline

# Ejecutar demostración
pipeline_demo = demo_advanced_pipeline()
```

## Análisis Exploratorio Automatizado

### 1. EDA Automatizado con Scikit-learn

```python
from sklearn.datasets import load_iris, load_boston, make_classification
from scipy import stats
import seaborn as sns

class AutomatedEDA:
    """Análisis exploratorio de datos automatizado"""
    
    def __init__(self, df, target_column=None):
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column and target_column in self.numeric_columns:
            self.numeric_columns.remove(target_column)
        if target_column and target_column in self.categorical_columns:
            self.categorical_columns.remove(target_column)
    
    def basic_info(self):
        """Información básica del dataset"""
        print("=== INFORMACIÓN BÁSICA DEL DATASET ===")
        print(f"Shape: {self.df.shape}")
        print(f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Valores nulos: {self.df.isnull().sum().sum()}")
        
        print("\nTipos de datos:")
        print(self.df.dtypes.value_counts())
        
        print("\nPrimeras 5 filas:")
        print(self.df.head())
        
        print("\nÚltimas 5 filas:")
        print(self.df.tail())
        
        return self.df.info()
    
    def missing_values_analysis(self):
        """Análisis de valores faltantes"""
        missing = self.df.isnull().sum()
        missing_percent = 100 * missing / len(self.df)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print("=== ANÁLISIS DE VALORES FALTANTES ===")
            print(missing_df)
            
            # Visualizar patrón de valores faltantes
            if len(missing_df) > 1:
                plt.figure(figsize=(12, 8))
                sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Patrón de Valores Faltantes')
                plt.show()
        else:
            print("=== No hay valores faltantes en el dataset ===")
        
        return missing_df
    
    def numerical_analysis(self):
        """Análisis de variables numéricas"""
        if not self.numeric_columns:
            print("No hay variables numéricas para analizar")
            return None
        
        print("=== ANÁLISIS DE VARIABLES NUMÉRICAS ===")
        
        # Estadísticas descriptivas
        desc_stats = self.df[self.numeric_columns].describe()
        print("\nEstadísticas descriptivas:")
        print(desc_stats)
        
        # Detectar outliers usando IQR
        outliers_info = []
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_info.append({
                'column': col,
                'outliers_count': len(outliers),
                'outliers_percentage': len(outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        
        outliers_df = pd.DataFrame(outliers_info)
        print("\nDetección de outliers (método IQR):")
        print(outliers_df[outliers_df['outliers_count'] > 0])
        
        # Pruebas de normalidad
        print("\nPruebas de normalidad (Shapiro-Wilk):")
        normality_tests = []
        for col in self.numeric_columns:
            if len(self.df[col].dropna()) > 3:  # Mínimo para Shapiro-Wilk
                stat, p_value = stats.shapiro(self.df[col].dropna()[:5000])  # Máximo 5000 muestras
                normality_tests.append({
                    'column': col,
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                })
        
        normality_df = pd.DataFrame(normality_tests)
        print(normality_df)
        
        # Visualizaciones
        self._plot_numerical_distributions()
        
        return {
            'descriptive_stats': desc_stats,
            'outliers': outliers_df,
            'normality_tests': normality_df
        }
    
    def _plot_numerical_distributions(self):
        """Crear visualizaciones para variables numéricas"""
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        # Histogramas
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
        
        # Ocultar subplots vacíos
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Boxplots para detectar outliers
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                self.df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Boxplot de {col}')
        
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_analysis(self):
        """Análisis de variables categóricas"""
        if not self.categorical_columns:
            print("No hay variables categóricas para analizar")
            return None
        
        print("=== ANÁLISIS DE VARIABLES CATEGÓRICAS ===")
        
        categorical_info = []
        for col in self.categorical_columns:
            unique_values = self.df[col].nunique()
            most_frequent = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else 'N/A'
            most_frequent_count = self.df[col].value_counts().iloc[0] if len(self.df[col]) > 0 else 0
            
            categorical_info.append({
                'column': col,
                'unique_values': unique_values,
                'most_frequent': most_frequent,
                'most_frequent_count': most_frequent_count,
                'most_frequent_percentage': most_frequent_count / len(self.df) * 100
            })
        
        cat_df = pd.DataFrame(categorical_info)
        print(cat_df)
        
        # Value counts para cada variable categórica
        for col in self.categorical_columns:
            print(f"\nDistribución de {col}:")
            value_counts = self.df[col].value_counts()
            print(value_counts.head(10))  # Top 10
            
            # Visualizar si no hay demasiadas categorías
            if len(value_counts) <= 10:
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar')
                plt.title(f'Distribución de {col}')
                plt.xlabel(col)
                plt.ylabel('Frecuencia')
                plt.xticks(rotation=45)
                plt.show()
        
        return cat_df
    
    def correlation_analysis(self):
        """Análisis de correlaciones"""
        if len(self.numeric_columns) < 2:
            print("Se necesitan al menos 2 variables numéricas para análisis de correlación")
            return None
        
        print("=== ANÁLISIS DE CORRELACIONES ===")
        
        # Matriz de correlación
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Encontrar correlaciones altas
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Correlación alta
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print("Correlaciones altas (|r| > 0.7):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.sort_values('correlation', key=abs, ascending=False))
        
        # Visualizar matriz de correlación
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.show()
        
        return corr_matrix
    
    def target_analysis(self):
        """Análisis específico de la variable objetivo"""
        if not self.target_column:
            print("No se especificó variable objetivo")
            return None
        
        print(f"=== ANÁLISIS DE VARIABLE OBJETIVO: {self.target_column} ===")
        
        target_series = self.df[self.target_column]
        
        if self.target_column in self.df.select_dtypes(include=[np.number]).columns:
            # Variable objetivo numérica
            print("Estadísticas de la variable objetivo:")
            print(target_series.describe())
            
            # Distribución
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            target_series.hist(bins=30, alpha=0.7)
            plt.title(f'Distribución de {self.target_column}')
            plt.xlabel(self.target_column)
            plt.ylabel('Frecuencia')
            
            plt.subplot(1, 2, 2)
            target_series.plot(kind='box')
            plt.title(f'Boxplot de {self.target_column}')
            
            plt.tight_layout()
            plt.show()
            
        else:
            # Variable objetivo categórica
            print("Distribución de clases:")
            class_distribution = target_series.value_counts()
            print(class_distribution)
            print(f"\nBalance de clases:")
            print(class_distribution / len(target_series) * 100)
            
            # Visualizar distribución
            plt.figure(figsize=(10, 6))
            class_distribution.plot(kind='bar')
            plt.title(f'Distribución de {self.target_column}')
            plt.xlabel('Clases')
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            plt.show()
        
        return target_series.describe() if target_series.dtype in [np.number] else class_distribution
    
    def generate_full_report(self):
        """Generar reporte completo de EDA"""
        print("GENERANDO REPORTE COMPLETO DE EDA...")
        print("=" * 50)
        
        # Ejecutar todos los análisis
        basic_info = self.basic_info()
        missing_analysis = self.missing_values_analysis()
        numerical_analysis = self.numerical_analysis()
        categorical_analysis = self.categorical_analysis()
        correlation_analysis = self.correlation_analysis()
        target_analysis = self.target_analysis()
        
        return {
            'basic_info': basic_info,
            'missing_values': missing_analysis,
            'numerical_analysis': numerical_analysis,
            'categorical_analysis': categorical_analysis,
            'correlations': correlation_analysis,
            'target_analysis': target_analysis
        }

# Ejemplo de uso del EDA automatizado
def demo_automated_eda():
    """Demostración del EDA automatizado"""
    
    # Crear dataset sintético más realista
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        flip_y=0.1,  # Añadir algo de ruido
        random_state=42
    )
    
    # Crear DataFrame con nombres más realistas
    feature_names = [
        'income', 'age', 'education_years', 'experience', 'debt_ratio',
        'savings', 'credit_score', 'loan_amount', 'property_value', 'employment_duration',
        'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['approved'] = y  # Variable objetivo binaria
    
    # Añadir algunas variables categóricas
    df['education_level'] = pd.cut(df['education_years'], 
                                  bins=[0, 12, 16, 20], 
                                  labels=['High School', 'Bachelor', 'Graduate'])
    df['income_bracket'] = pd.cut(df['income'], 
                                 bins=5, 
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Introducir algunos valores faltantes de manera realista
    np.random.seed(42)
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices, 'savings'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'education_level'] = np.nan
    
    print("Dataset creado:")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Ejecutar EDA automatizado
    eda = AutomatedEDA(df, target_column='approved')
    report = eda.generate_full_report()
    
    return eda, report

# Ejecutar demostración del EDA
eda_demo, eda_report = demo_automated_eda()
```

## Clustering y Reducción de Dimensionalidad

### 1. Análisis de Clusters Avanzado

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

class AdvancedClusteringAnalysis:
    """Análisis avanzado de clustering con múltiples algoritmos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_data(self, X):
        """Preparar datos para clustering"""
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Encontrar número óptimo de clusters usando múltiples métricas"""
        
        X_scaled = self.prepare_data(X)
        
        # Métricas para evaluar
        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': []
        }
        
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calcular métricas
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(X_scaled, cluster_labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, cluster_labels))
        
        # Visualizar métricas
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow method (Inertia)
        axes[0].plot(cluster_range, metrics['inertia'], 'bo-')
        axes[0].set_xlabel('Número de Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        # Silhouette Score
        axes[1].plot(cluster_range, metrics['silhouette'], 'ro-')
        axes[1].set_xlabel('Número de Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True)
        
        # Calinski-Harabasz Index
        axes[2].plot(cluster_range, metrics['calinski_harabasz'], 'go-')
        axes[2].set_xlabel('Número de Clusters')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Encontrar número óptimo basado en silhouette score
        optimal_clusters = cluster_range[np.argmax(metrics['silhouette'])]
        print(f"Número óptimo de clusters (Silhouette): {optimal_clusters}")
        
        self.results['optimal_analysis'] = {
            'cluster_range': list(cluster_range),
            'metrics': metrics,
            'optimal_clusters': optimal_clusters
        }
        
        return optimal_clusters, metrics
    
    def compare_clustering_algorithms(self, X, n_clusters=3):
        """Comparar diferentes algoritmos de clustering"""
        
        X_scaled = self.prepare_data(X)
        
        # Definir algoritmos
        algorithms = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42),
            'Gaussian Mixture': GaussianMixture(n_components=n_clusters, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Ejecutando {name}...")
            
            if name == 'Gaussian Mixture':
                cluster_labels = algorithm.fit_predict(X_scaled)
            else:
                cluster_labels = algorithm.fit_predict(X_scaled)
            
            # Calcular métricas
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters_found > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
            else:
                silhouette = -1
                calinski_harabasz = -1
            
            results[name] = {
                'labels': cluster_labels,
                'n_clusters': n_clusters_found,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'algorithm': algorithm
            }
        
        # Mostrar comparación
        comparison_df = pd.DataFrame([
            {
                'Algorithm': name,
                'Clusters_Found': results[name]['n_clusters'],
                'Silhouette_Score': results[name]['silhouette_score'],
                'Calinski_Harabasz_Score': results[name]['calinski_harabasz_score']
            }
            for name in results.keys()
        ])
        
        print("\nComparación de Algoritmos:")
        print(comparison_df.sort_values('Silhouette_Score', ascending=False))
        
        self.results['algorithm_comparison'] = results
        return results
    
    def visualize_clusters_2d(self, X, clustering_results):
        """Visualizar clusters en 2D usando PCA y t-SNE"""
        
        X_scaled = self.prepare_data(X)
        
        # Reducción de dimensionalidad
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE (solo si tenemos suficientes muestras)
        if X_scaled.shape[0] > 30:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_scaled.shape[0]-1))
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = None
        
        # UMAP (si está disponible)
        try:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = umap_reducer.fit_transform(X_scaled)
        except:
            X_umap = None
        
        # Crear visualizaciones
        n_plots = 2 + (1 if X_tsne is not None else 0) + (1 if X_umap is not None else 0)
        n_algorithms = len(clustering_results)
        
        fig, axes = plt.subplots(n_algorithms, n_plots, figsize=(5*n_plots, 4*n_algorithms))
        if n_algorithms == 1:
            axes = axes.reshape(1, -1)
        
        for i, (alg_name, result) in enumerate(clustering_results.items()):
            labels = result['labels']
            
            # PCA
            scatter = axes[i, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
            axes[i, 0].set_title(f'{alg_name} - PCA')
            axes[i, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[i, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # Original space (si es 2D)
            if X.shape[1] == 2:
                axes[i, 1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7)
                axes[i, 1].set_title(f'{alg_name} - Original Space')
            else:
                axes[i, 1].text(0.5, 0.5, 'Original space\n(>2D)', ha='center', va='center',
                              transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'{alg_name} - Original Space')
            
            plot_idx = 2
            
            # t-SNE
            if X_tsne is not None:
                axes[i, plot_idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
                axes[i, plot_idx].set_title(f'{alg_name} - t-SNE')
                plot_idx += 1
            
            # UMAP
            if X_umap is not None:
                axes[i, plot_idx].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', alpha=0.7)
                axes[i, plot_idx].set_title(f'{alg_name} - UMAP')
        
        plt.tight_layout()
        plt.show()
        
        return X_pca, X_tsne, X_umap
    
    def cluster_profiling(self, X, labels, feature_names=None):
        """Crear perfil de cada cluster"""
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = labels
        
        # Estadísticas por cluster
        cluster_profiles = df.groupby('cluster').agg(['mean', 'std', 'count'])
        
        print("=== PERFILES DE CLUSTERS ===")
        print(cluster_profiles)
        
        # Identificar características distintivas de cada cluster
        distinctive_features = {}
        
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:  # Noise in DBSCAN
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            other_data = df[df['cluster'] != cluster_id]
            
            feature_importance = []
            for feature in feature_names:
                cluster_mean = cluster_data[feature].mean()
                other_mean = other_data[feature].mean()
                
                # Calcular diferencia normalizada
                difference = abs(cluster_mean - other_mean)
                overall_std = df[feature].std()
                
                if overall_std > 0:
                    normalized_diff = difference / overall_std
                    feature_importance.append({
                        'feature': feature,
                        'cluster_mean': cluster_mean,
                        'others_mean': other_mean,
                        'difference': difference,
                        'normalized_difference': normalized_diff
                    })
            
            # Ordenar por importancia
            feature_importance = sorted(feature_importance, 
                                     key=lambda x: x['normalized_difference'], 
                                     reverse=True)
            
            distinctive_features[cluster_id] = feature_importance[:5]  # Top 5
        
        # Mostrar características distintivas
        print("\n=== CARACTERÍSTICAS DISTINTIVAS POR CLUSTER ===")
        for cluster_id, features in distinctive_features.items():
            print(f"\nCluster {cluster_id}:")
            for feature_info in features:
                print(f"  {feature_info['feature']}: "
                      f"Cluster={feature_info['cluster_mean']:.2f}, "
                      f"Others={feature_info['others_mean']:.2f}, "
                      f"Diff={feature_info['normalized_difference']:.2f}")
        
        return cluster_profiles, distinctive_features

# Demostración completa de clustering
def demo_advanced_clustering():
    """Demostración del análisis avanzado de clustering"""
    
    # Crear dataset con clusters naturales
    from sklearn.datasets import make_blobs
    
    X, true_labels = make_blobs(
        n_samples=300,
        centers=4,
        n_features=8,
        cluster_std=1.5,
        random_state=42
    )
    
    feature_names = [f'dimension_{i+1}' for i in range(X.shape[1])]
    
    print("=== ANÁLISIS AVANZADO DE CLUSTERING ===")
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} dimensiones")
    
    # Crear analizador
    clustering_analyzer = AdvancedClusteringAnalysis()
    
    # 1. Encontrar número óptimo de clusters
    print("\n1. Búsqueda del número óptimo de clusters:")
    optimal_k, metrics = clustering_analyzer.find_optimal_clusters(X, max_clusters=8)
    
    # 2. Comparar algoritmos
    print(f"\n2. Comparación de algoritmos con {optimal_k} clusters:")
    results = clustering_analyzer.compare_clustering_algorithms(X, n_clusters=optimal_k)
    
    # 3. Visualizar resultados
    print("\n3. Visualización de resultados:")
    visualizations = clustering_analyzer.visualize_clusters_2d(X, results)
    
    # 4. Perfilar clusters del mejor algoritmo
    best_algorithm = max(results.keys(), 
                        key=lambda x: results[x]['silhouette_score'] if results[x]['silhouette_score'] > 0 else -1)
    
    print(f"\n4. Perfilando clusters del mejor algoritmo: {best_algorithm}")
    best_labels = results[best_algorithm]['labels']
    
    profiles, distinctive = clustering_analyzer.cluster_profiling(X, best_labels, feature_names)
    
    # 5. Comparar con clusters verdaderos (si están disponibles)
    print(f"\n5. Comparación con clusters verdaderos:")
    if true_labels is not None:
        ari_score = adjusted_rand_score(true_labels, best_labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        
        # Visualizar comparación
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PCA para visualización
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
        axes[0].set_title('Clusters Verdaderos')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='tab10', alpha=0.7)
        axes[1].set_title(f'Clusters Detectados ({best_algorithm})')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        
        plt.tight_layout()
        plt.show()
    
    return clustering_analyzer

# Ejecutar demostración de clustering
clustering_demo = demo_advanced_clustering()
```

## Conclusiones Scikit-learn 2024

Las fortalezas perdurables de scikit-learn:

🏗️ **API consistente**: Fit/predict/transform unificado para todos los algoritmos  
⚡ **Eficiencia probada**: Optimización C/Cython para performance  
📚 **Algoritmos clásicos**: Implementaciones robustas y bien documentadas  
🔧 **Pipelines potentes**: Automatización completa del flujo ML  
📊 **Interpretabilidad**: Modelos explicables para decisiones críticas  
🧪 **Validación rigurosa**: Cross-validation y métricas comprensivas  

Scikit-learn sigue siendo indispensable porque:
- **Resuelve el 80% de problemas** de ML empresarial
- **Baseline rápido** para cualquier proyecto
- **Debugging simplificado** vs deep learning
- **Deployment ligero** sin dependencias pesadas
- **Explicabilidad natural** para stakeholders

---
*¿Sigues usando scikit-learn en 2024? Comparte qué algoritmos te han dado mejores resultados en producción.*