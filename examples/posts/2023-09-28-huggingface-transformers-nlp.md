---
layout: post
title: Hugging Face Transformers - El Ecosistema NLP Definitivo
tags: [huggingface, transformers, nlp, bert, gpt, machine-learning, python]
---

**Hugging Face** se ha convertido en el ecosistema definitivo para Natural Language Processing. Con su biblioteca Transformers y el Hub de modelos preentrenados, democratiza el acceso a los modelos de lenguaje más avanzados del mundo.

## Introducción al Ecosistema Hugging Face

### 1. Configuración y Primeros Pasos

```python
# Instalación completa del ecosistema
# pip install transformers datasets tokenizers accelerate evaluate
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Verificar configuración
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2. Pipelines - La Forma Más Rápida de Empezar

```python
from transformers import pipeline

# Análisis de sentimientos
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

# Ejemplos en español
textos_ejemplo = [
    "Me encanta este producto, es fantástico!",
    "El servicio fue terrible, muy decepcionante.",
    "No está mal, pero podría mejorar.",
    "Excelente calidad precio, lo recomiendo totalmente."
]

resultados = sentiment_analyzer(textos_ejemplo)
for texto, resultado in zip(textos_ejemplo, resultados):
    print(f"Texto: {texto}")
    print(f"Sentimiento: {resultado['label']} (confianza: {resultado['score']:.3f})")
    print("-" * 50)

# Generación de texto
generador = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    max_length=50,
    num_return_sequences=2,
    temperature=0.8,
    device=0 if torch.cuda.is_available() else -1
)

prompt = "El futuro de la inteligencia artificial"
generaciones = generador(prompt, max_length=100, num_return_sequences=3)

print("Generaciones de texto:")
for i, gen in enumerate(generaciones):
    print(f"{i+1}: {gen['generated_text']}")
    print()

# Question Answering
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0 if torch.cuda.is_available() else -1
)

contexto = """
La inteligencia artificial (IA) es una rama de las ciencias de la computación que se encarga del 
estudio y desarrollo de sistemas capaces de realizar tareas que requieren inteligencia humana. 
Estos sistemas pueden aprender, razonar, percibir, procesar lenguaje natural y tomar decisiones. 
El machine learning es una subdisciplina de la IA que permite a las máquinas aprender 
automáticamente sin ser programadas explícitamente.
"""

preguntas = [
    "¿Qué es la inteligencia artificial?",
    "¿Qué puede hacer un sistema de IA?",
    "¿Cómo se relaciona el machine learning con la IA?"
]

for pregunta in preguntas:
    respuesta = qa_pipeline(question=pregunta, context=contexto)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {respuesta['answer']}")
    print(f"Confianza: {respuesta['score']:.3f}")
    print("-" * 50)
```

## Modelos Preentrenados y Fine-tuning

### 1. Clasificación de Texto Personalizada

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class CustomTextClassifier:
    """Clasificador personalizado usando Hugging Face"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Configurar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_data(self, texts, labels=None, max_length=512):
        """Preparar datos para entrenamiento/inferencia"""
        
        # Tokenizar textos
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if labels is not None:
            dataset_dict['labels'] = torch.tensor(labels, dtype=torch.long)
            
        return Dataset.from_dict(dataset_dict)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Entrenar el modelo"""
        
        # Preparar datasets
        train_dataset = self.prepare_data(train_texts, train_labels)
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = self.prepare_data(val_texts, val_labels)
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            learning_rate=learning_rate,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Función de métricas
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            accuracy = accuracy_score(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        # Entrenar
        print("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo final
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Entrenamiento completado!")
        
        return trainer
    
    def predict(self, texts, batch_size=16):
        """Realizar predicciones"""
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Procesar en batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Predicción
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                predictions.extend(batch_predictions)
                probabilities.extend(batch_probs)
        
        return predictions, probabilities

# Ejemplo de uso con datos sintéticos
def demo_custom_classifier():
    """Demostración del clasificador personalizado"""
    
    # Crear datos sintéticos de ejemplo (reviews de productos)
    positive_reviews = [
        "Este producto es increíble, superó mis expectativas",
        "Excelente calidad, lo recomiendo totalmente",
        "Muy satisfecho con la compra, llegó rápido",
        "Fantástico, justo lo que necesitaba",
        "Producto de alta calidad, vale la pena el precio"
    ] * 50  # Duplicar para tener más datos
    
    negative_reviews = [
        "Producto defectuoso, no funciona correctamente",
        "Muy decepcionante, no vale el dinero",
        "Mala calidad, se rompió al primer uso",
        "No lo recomiendo, servicio terrible",
        "Perdida de dinero, no compren este producto"
    ] * 50
    
    # Combinar datos
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    # Split train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Crear y entrenar clasificador
    classifier = CustomTextClassifier(
        model_name="distilbert-base-uncased",
        num_labels=2
    )
    
    # Entrenar (con un subset pequeño para demo)
    trainer = classifier.train(
        train_texts[:100],  # Solo 100 ejemplos para demo rápida
        train_labels[:100],
        val_texts=test_texts[:20],
        val_labels=test_labels[:20],
        num_epochs=2,
        batch_size=8
    )
    
    # Evaluar
    predictions, probabilities = classifier.predict(test_texts[:10])
    
    print("\nResultados de predicción:")
    label_names = ["Negativo", "Positivo"]
    
    for i, (text, pred, prob, true_label) in enumerate(
        zip(test_texts[:10], predictions, probabilities[:10], test_labels[:10])
    ):
        print(f"\nTexto {i+1}: {text[:100]}...")
        print(f"Predicción: {label_names[pred]} (confianza: {prob[pred]:.3f})")
        print(f"Real: {label_names[true_label]}")
        print(f"Correcto: {'✓' if pred == true_label else '✗'}")

# Ejecutar demo (descomenta para probar)
# demo_custom_classifier()
```

### 2. Generación de Texto Avanzada

```python
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TextGenerationPipeline,
    StoppingCriteria,
    StoppingCriteriaList
)

class AdvancedTextGenerator:
    """Generador de texto avanzado con control fino"""
    
    def __init__(self, model_name="gpt2-medium"):
        self.model_name = model_name
        
        # Cargar modelo y tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Configurar pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Mover a GPU si está disponible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Crear pipeline
        self.generator = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def generate_with_control(self, prompt, max_length=100, temperature=0.8, 
                             top_p=0.9, repetition_penalty=1.1, num_return_sequences=1,
                             do_sample=True, early_stopping=True):
        """Generar texto con control avanzado de parámetros"""
        
        # Configurar stopping criteria personalizada
        class CustomStoppingCriteria(StoppingCriteria):
            def __init__(self, stop_words, tokenizer):
                self.stop_words = stop_words
                self.tokenizer = tokenizer
                
            def __call__(self, input_ids, scores, **kwargs):
                # Convertir tokens a texto
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
                # Verificar si contiene alguna palabra de parada
                return any(stop_word in text.lower() for stop_word in self.stop_words)
        
        # Palabras de parada personalizadas
        stop_words = ["the end", "conclusion", "final"]
        stopping_criteria = StoppingCriteriaList([
            CustomStoppingCriteria(stop_words, self.tokenizer)
        ])
        
        # Generar texto
        results = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return results
    
    def generate_creative_writing(self, prompt, style="narrative"):
        """Generar escritura creativa con estilos específicos"""
        
        style_prompts = {
            "narrative": f"Here's a compelling story: {prompt}",
            "technical": f"Technical explanation: {prompt}",
            "poetic": f"In poetic form: {prompt}",
            "academic": f"Academic analysis: {prompt}",
            "conversational": f"Let me tell you about {prompt}"
        }
        
        enhanced_prompt = style_prompts.get(style, prompt)
        
        # Parámetros optimizados por estilo
        style_params = {
            "narrative": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1},
            "technical": {"temperature": 0.3, "top_p": 0.7, "repetition_penalty": 1.2},
            "poetic": {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.0},
            "academic": {"temperature": 0.4, "top_p": 0.8, "repetition_penalty": 1.15},
            "conversational": {"temperature": 0.7, "top_p": 0.85, "repetition_penalty": 1.1}
        }
        
        params = style_params.get(style, {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1})
        
        return self.generate_with_control(
            enhanced_prompt,
            max_length=200,
            **params,
            num_return_sequences=3
        )
    
    def interactive_generation(self, initial_prompt):
        """Generación interactiva con el usuario"""
        
        current_text = initial_prompt
        conversation_history = [current_text]
        
        print("=== Generación Interactiva ===")
        print("Escribe 'quit' para salir, 'regenerate' para regenerar la última respuesta")
        print(f"\nTexto inicial: {current_text}")
        
        while True:
            user_input = input("\n¿Continuar con qué? (o 'quit'/'regenerate'): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'regenerate':
                if len(conversation_history) > 1:
                    current_text = conversation_history[-2]
                    conversation_history = conversation_history[:-1]
                else:
                    current_text = initial_prompt
            else:
                current_text += f" {user_input}"
            
            # Generar continuación
            print("\nGenerando...")
            results = self.generate_with_control(
                current_text,
                max_length=len(current_text.split()) + 50,
                temperature=0.8,
                num_return_sequences=2
            )
            
            print("\nOpciones generadas:")
            for i, result in enumerate(results):
                generated_part = result['generated_text'][len(current_text):].strip()
                print(f"{i+1}. {generated_part}")
            
            # Selección del usuario
            while True:
                try:
                    choice = input("\n¿Cuál eliges? (1-2): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(results):
                        current_text = results[choice_idx]['generated_text']
                        conversation_history.append(current_text)
                        print(f"\nTexto actualizado: {current_text}")
                        break
                    else:
                        print("Opción inválida. Intenta de nuevo.")
                except ValueError:
                    print("Por favor ingresa un número válido.")
        
        print("\n=== Conversación Final ===")
        print(current_text)

# Ejemplo de uso del generador avanzado
def demo_advanced_generator():
    """Demostración del generador avanzado"""
    
    generator = AdvancedTextGenerator("gpt2")
    
    # Generación básica
    print("=== Generación Básica ===")
    prompt = "The future of artificial intelligence"
    results = generator.generate_with_control(
        prompt,
        max_length=150,
        temperature=0.8,
        num_return_sequences=2
    )
    
    for i, result in enumerate(results):
        print(f"\nVariación {i+1}:")
        print(result['generated_text'])
    
    # Generación por estilos
    print("\n=== Generación por Estilos ===")
    topic = "machine learning algorithms"
    
    styles = ["narrative", "technical", "poetic"]
    for style in styles:
        print(f"\n--- Estilo: {style} ---")
        results = generator.generate_creative_writing(topic, style)
        print(results[0]['generated_text'])

# Ejecutar demo (descomenta para probar)
# demo_advanced_generator()
```

## Modelos Multimodales y Avanzados

### 1. CLIP - Vision y Lenguaje

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class CLIPAnalyzer:
    """Analizador multimodal con CLIP"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def load_image_from_url(self, url):
        """Cargar imagen desde URL"""
        try:
            response = requests.get(url)
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return None
    
    def analyze_image_text_similarity(self, image, text_candidates):
        """Analizar similitud entre imagen y textos candidatos"""
        
        if isinstance(image, str):  # Si es URL
            image = self.load_image_from_url(image)
            
        if image is None:
            return None
        
        # Procesar inputs
        inputs = self.processor(
            text=text_candidates,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Normalizar embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            # Calcular similitudes
            similarities = torch.matmul(image_embeds, text_embeds.T)
            
        return similarities.cpu().numpy().flatten()
    
    def find_best_caption(self, image, captions):
        """Encontrar la mejor descripción para una imagen"""
        
        similarities = self.analyze_image_text_similarity(image, captions)
        if similarities is None:
            return None
            
        best_idx = np.argmax(similarities)
        
        results = []
        for i, (caption, similarity) in enumerate(zip(captions, similarities)):
            results.append({
                'caption': caption,
                'similarity': float(similarity),
                'is_best': i == best_idx
            })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def classify_image(self, image, class_names):
        """Clasificar imagen usando nombres de clases"""
        
        # Crear prompts de clasificación
        text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
        
        similarities = self.analyze_image_text_similarity(image, text_prompts)
        if similarities is None:
            return None
        
        # Aplicar softmax para obtener probabilidades
        probabilities = torch.softmax(torch.tensor(similarities), dim=0).numpy()
        
        results = []
        for class_name, prob in zip(class_names, probabilities):
            results.append({
                'class': class_name,
                'probability': float(prob)
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)
    
    def visual_search(self, query_image, candidate_images, descriptions=None):
        """Búsqueda visual: encontrar imágenes similares"""
        
        if isinstance(query_image, str):
            query_image = self.load_image_from_url(query_image)
            
        # Procesar imagen de consulta
        query_inputs = self.processor(
            images=query_image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            query_embeds = self.model.get_image_features(**query_inputs)
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        
        results = []
        
        for i, candidate in enumerate(candidate_images):
            if isinstance(candidate, str):
                candidate = self.load_image_from_url(candidate)
            
            if candidate is None:
                continue
                
            # Procesar imagen candidata
            candidate_inputs = self.processor(
                images=candidate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                candidate_embeds = self.model.get_image_features(**candidate_inputs)
                candidate_embeds = candidate_embeds / candidate_embeds.norm(dim=-1, keepdim=True)
                
                # Calcular similitud
                similarity = torch.matmul(query_embeds, candidate_embeds.T).item()
            
            result = {
                'index': i,
                'similarity': similarity,
                'image': candidate
            }
            
            if descriptions and i < len(descriptions):
                result['description'] = descriptions[i]
                
            results.append(result)
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Ejemplo de uso de CLIP
def demo_clip_analyzer():
    """Demostración del analizador CLIP"""
    
    analyzer = CLIPAnalyzer()
    
    # URLs de imágenes de ejemplo
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Meow.jpg/256px-Meow.jpg",  # Gato
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/256px-Golde33443.jpg",  # Perro
    ]
    
    # Clasificación de imagen
    print("=== Clasificación de Imagen ===")
    if image_urls:
        image = analyzer.load_image_from_url(image_urls[0])
        if image:
            classes = ["cat", "dog", "bird", "car", "tree", "house"]
            results = analyzer.classify_image(image, classes)
            
            print("Clasificación:")
            for result in results[:3]:
                print(f"  {result['class']}: {result['probability']:.3f}")
    
    # Generación de captions
    print("\n=== Selección de Caption ===")
    captions = [
        "a cute cat sitting on a chair",
        "a dog running in the park", 
        "a bird flying in the sky",
        "a car driving on the road",
        "a beautiful landscape"
    ]
    
    if image_urls:
        image = analyzer.load_image_from_url(image_urls[0])
        if image:
            results = analyzer.find_best_caption(image, captions)
            
            print("Mejores captions:")
            for result in results[:3]:
                print(f"  '{result['caption']}': {result['similarity']:.3f}")

# Ejecutar demo CLIP (descomenta para probar)
# demo_clip_analyzer()
```

### 2. Modelos de Embeddings Especializados

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class AdvancedEmbeddingAnalyzer:
    """Analizador avanzado de embeddings semánticos"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Usar Sentence Transformers para embeddings optimizados
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_embeddings(self, texts):
        """Generar embeddings para una lista de textos"""
        return self.model.encode(texts, convert_to_tensor=True, device=self.device)
    
    def semantic_search(self, query, corpus, top_k=5):
        """Búsqueda semántica en un corpus de documentos"""
        
        # Generar embeddings
        query_embedding = self.generate_embeddings([query])
        corpus_embeddings = self.generate_embeddings(corpus)
        
        # Calcular similitudes
        similarities = torch.cosine_similarity(query_embedding, corpus_embeddings)
        
        # Obtener top-k resultados
        top_indices = torch.topk(similarities, k=min(top_k, len(corpus))).indices
        
        results = []
        for idx in top_indices:
            results.append({
                'text': corpus[idx.item()],
                'similarity': similarities[idx.item()].item(),
                'index': idx.item()
            })
        
        return results
    
    def cluster_texts(self, texts, n_clusters=5, random_state=42):
        """Agrupar textos por similitud semántica"""
        
        # Generar embeddings
        embeddings = self.generate_embeddings(texts).cpu().numpy()
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organizar resultados por cluster
        clusters = {}
        for i, (text, label) in enumerate(zip(texts, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'text': text,
                'index': i
            })
        
        return clusters, embeddings, cluster_labels
    
    def visualize_embeddings(self, texts, labels=None, title="Embeddings Visualization"):
        """Visualizar embeddings usando t-SNE"""
        
        # Generar embeddings
        embeddings = self.generate_embeddings(texts).cpu().numpy()
        
        # Aplicar t-SNE para reducir dimensionalidad
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[color], label=f'Cluster {label}', alpha=0.7, s=50)
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
        
        plt.title(title)
        if labels is not None:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return embeddings_2d
    
    def find_outliers(self, texts, threshold=0.3):
        """Encontrar textos atípicos (outliers) en el conjunto"""
        
        embeddings = self.generate_embeddings(texts).cpu().numpy()
        
        # Calcular similitud promedio de cada texto con todos los demás
        similarities = cosine_similarity(embeddings)
        
        # Calcular score de similitud promedio (excluyendo self-similarity)
        avg_similarities = []
        for i in range(len(similarities)):
            # Excluir la diagonal (similitud consigo mismo = 1.0)
            mask = np.ones(len(similarities), dtype=bool)
            mask[i] = False
            avg_sim = np.mean(similarities[i][mask])
            avg_similarities.append(avg_sim)
        
        # Identificar outliers
        outliers = []
        for i, avg_sim in enumerate(avg_similarities):
            if avg_sim < threshold:
                outliers.append({
                    'text': texts[i],
                    'index': i,
                    'avg_similarity': avg_sim
                })
        
        return sorted(outliers, key=lambda x: x['avg_similarity'])
    
    def semantic_similarity_matrix(self, texts):
        """Crear matriz de similitud semántica"""
        
        embeddings = self.generate_embeddings(texts).cpu().numpy()
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def paraphrase_mining(self, texts, threshold=0.7):
        """Encontrar posibles paráfrasis en el conjunto de textos"""
        
        similarity_matrix = self.semantic_similarity_matrix(texts)
        
        paraphrases = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    paraphrases.append({
                        'text1': texts[i],
                        'text2': texts[j],
                        'similarity': similarity,
                        'indices': (i, j)
                    })
        
        return sorted(paraphrases, key=lambda x: x['similarity'], reverse=True)

# Ejemplo completo de análisis de embeddings
def demo_embedding_analyzer():
    """Demostración completa del analizador de embeddings"""
    
    analyzer = AdvancedEmbeddingAnalyzer()
    
    # Corpus de ejemplo sobre tecnología
    tech_texts = [
        "Machine learning algorithms can predict customer behavior",
        "Artificial intelligence is transforming healthcare industry",
        "Deep learning models require large datasets for training",
        "Natural language processing enables chatbot conversations",
        "Computer vision systems can detect objects in images",
        "Blockchain technology ensures secure financial transactions",
        "Cloud computing provides scalable infrastructure solutions",
        "Cybersecurity protects against digital threats and attacks",
        "Data science extracts insights from large datasets", 
        "Quantum computing promises exponential processing power",
        "My cat loves to play with yarn balls",  # Outlier
        "The weather today is quite pleasant"     # Outlier
    ]
    
    print("=== Análisis de Embeddings ===")
    print(f"Procesando {len(tech_texts)} textos...")
    
    # 1. Búsqueda semántica
    print("\n1. Búsqueda Semántica:")
    query = "AI and neural networks"
    search_results = analyzer.semantic_search(query, tech_texts, top_k=3)
    
    print(f"Query: '{query}'")
    for result in search_results:
        print(f"  Similitud {result['similarity']:.3f}: {result['text']}")
    
    # 2. Clustering
    print("\n2. Clustering de Textos:")
    clusters, embeddings, labels = analyzer.cluster_texts(tech_texts, n_clusters=3)
    
    for cluster_id, texts in clusters.items():
        print(f"\nCluster {cluster_id}:")
        for text_info in texts:
            print(f"  - {text_info['text']}")
    
    # 3. Detección de outliers
    print("\n3. Detección de Outliers:")
    outliers = analyzer.find_outliers(tech_texts, threshold=0.4)
    
    if outliers:
        print("Textos atípicos encontrados:")
        for outlier in outliers:
            print(f"  Similitud promedio {outlier['avg_similarity']:.3f}: {outlier['text']}")
    else:
        print("No se encontraron outliers significativos")
    
    # 4. Minería de paráfrasis
    print("\n4. Búsqueda de Paráfrasis:")
    paraphrases = analyzer.paraphrase_mining(tech_texts, threshold=0.6)
    
    if paraphrases:
        print("Posibles paráfrasis encontradas:")
        for para in paraphrases[:3]:  # Solo top 3
            print(f"  Similitud {para['similarity']:.3f}:")
            print(f"    '{para['text1']}'")
            print(f"    '{para['text2']}'")
    else:
        print("No se encontraron paráfrasis con el threshold especificado")
    
    # 5. Visualización (opcional, requiere matplotlib)
    try:
        print("\n5. Generando visualización...")
        analyzer.visualize_embeddings(tech_texts, labels, 
                                    title="Clustering de Textos Tecnológicos")
    except Exception as e:
        print(f"No se pudo generar visualización: {e}")

# Ejecutar demo completo (descomenta para probar)
# demo_embedding_analyzer()
```

## Conclusiones del Ecosistema Hugging Face

Las ventajas clave incluyen:

🤖 **Modelos preentrenados**: Miles de modelos state-of-the-art listos para usar  
🔧 **Fine-tuning simplificado**: Pipelines optimizados para personalización  
🌐 **Multimodalidad**: CLIP, DALL-E, modelos vision-language  
⚡ **Performance**: Optimizaciones automáticas y aceleración GPU  
📚 **Datasets Hub**: Acceso a datasets curados y estandarizados  
🔄 **Interoperabilidad**: Compatible con PyTorch, TensorFlow, JAX  

Hugging Face ha democratizado el acceso a modelos de IA avanzados, permitiendo que cualquier desarrollador pueda implementar soluciones de NLP y multimodales de nivel empresarial con pocas líneas de código.

---
*¿Has usado Hugging Face en tus proyectos? Comparte qué modelos te han funcionado mejor en los comentarios.*