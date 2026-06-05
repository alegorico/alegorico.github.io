---
layout: post
title: FastAPI vs Flask - Batalla de Frameworks Python 2025
tags: [python, fastapi, flask, api, async, performance, web-development]
---

En 2025, la elección entre **FastAPI** y **Flask** sigue siendo uno de los debates más candentes en el ecosistema Python. Como desarrollador que ha trabajado extensivamente con ambos frameworks, incluyendo proyectos como [Flask-WebApi-demo](https://github.com/alegorico/Flask-WebApi-demo) (un gestor de contactos con arquitectura modular), analicemos cuándo usar cada uno y por qué FastAPI está ganando terreno.

## TL;DR - ¿Cuál elegir?

- **Flask**: Proyectos simples, máximo control, ecosistema maduro
- **FastAPI**: APIs modernas, async nativo, documentación automática

## FastAPI: El futuro es async

### ¿Por qué FastAPI domina en 2025?

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="Mi API", version="2.0.0")

class UserCreate(BaseModel):
    name: str
    email: str
    age: Optional[int] = None

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, background_tasks: BackgroundTasks):
    # Validación automática con Pydantic
    new_user = await save_user_async(user)
    
    # Tareas en background
    background_tasks.add_task(send_welcome_email, user.email)
    
    return new_user

@app.get("/users/{user_id}")
async def get_user(user_id: int) -> User:
    # Type hints automáticos para OpenAPI
    return await fetch_user_async(user_id)
```

### Ventajas clave de FastAPI

**1. Performance superior**
```python
# Comparativa de throughput (requests/segundo)
# FastAPI: ~25,000 req/s
# Flask + Gunicorn: ~8,000 req/s  
# Django: ~3,000 req/s

# Async nativo
async def heavy_computation():
    await asyncio.sleep(1)  # No bloquea otros requests
    return await process_data()
```

**2. Documentación automática**
```python
@app.get("/items/{item_id}")
async def read_item(
    item_id: int, 
    q: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    """
    Retrieve an item by ID.
    
    - **item_id**: The ID of the item to retrieve
    - **q**: Optional search query
    - **limit**: Number of results (1-100)
    """
    return {"item_id": item_id, "q": q, "limit": limit}

# Swagger UI automático en /docs
# ReDoc automático en /redoc
```

**3. Validación robusta**
```python
from pydantic import BaseModel, validator, Field
from datetime import datetime
from typing import List

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('name')
    def name_must_be_alphanumeric(cls, v):
        assert v.replace(' ', '').isalnum(), 'Name must be alphanumeric'
        return v.title()

# Errores automáticos con status 422
```

## Flask: La flexibilidad que nunca pasa de moda

### Mi Experiencia Práctica con Flask

He desarrollado múltiples proyectos con Flask, siendo el más representativo [Flask-WebApi-demo](https://github.com/alegorico/Flask-WebApi-demo), un gestor de contactos que demuestra patrones modernos de Flask:

**Características del proyecto:**
- **Arquitectura modular** con blueprints (`config/` y `generador/`)
- **API RESTful** completa en `/api/v1/contactos`
- **Frontend integrado** que consume la API con JavaScript
- **Configuración por entornos** para desarrollo/producción
- **Gestión de estado en memoria** ideal para prototipos

```python
# Estructura típica Flask - Flask-WebApi-demo
flask-webapi-demo/
├── config/           # Configuración por entornos
├── generador/        # Lógica de negocio modular
├── entrypoint.py     # Punto de entrada de la aplicación
└── requirements.txt  # Dependencias explícitas
```

El acceso principal es por `http://localhost:5000/gen/`, demostrando cómo Flask permite crear tanto APIs como interfaces web en el mismo proyecto con total flexibilidad.

### Cuándo Flask sigue siendo mejor

```python
from flask import Flask, request, jsonify
from functools import wraps
import jwt

app = Flask(__name__)

# Control total sobre middleware
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token required'}), 401
        
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/protected')
@auth_required
def protected_route():
    return jsonify({'message': f'Hello {request.user["username"]}'})
```

### Ventajas de Flask en 2025

**1. Ecosistema maduro**
```python
# Extensiones battle-tested
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf import FlaskForm
from flask_admin import Admin

# 15+ años de extensiones estables
```

**2. Microservicios simples**
```python
# Perfecto para servicios pequeños y específicos
app = Flask(__name__)

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    result = simple_processing(data)
    return {'result': result}

if __name__ == '__main__':
    app.run()
```

**3. Aprendizaje gradual**
```python
# Empezar simple, crecer orgánicamente
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

# Añadir complejidad progresivamente
```

## Comparativa práctica 2025

### Performance
```python
# Benchmark: API CRUD con 1000 requests concurrentes

# FastAPI + uvicorn
# GET /users: 24,000 req/s
# POST /users: 18,000 req/s
# Latencia p95: 15ms

# Flask + gunicorn (4 workers)  
# GET /users: 8,500 req/s
# POST /users: 6,200 req/s
# Latencia p95: 45ms
```

### Desarrollo
```python
# FastAPI: Más código automático
class UserAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/users/", response_model=User)
        async def create_user(user: UserCreate):
            # Validación automática
            # Documentación automática  
            # Type safety automático
            return await self.create_user_service(user)

# Flask: Más control manual
class UserAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/users/', methods=['POST'])
        def create_user():
            # Validación manual
            data = request.get_json()
            if not self.validate_user_data(data):
                return {'error': 'Invalid data'}, 400
            
            user = self.create_user_service(data)
            return jsonify(user.to_dict())
```

## Casos de uso específicos

### Elige FastAPI para:
- **APIs REST modernas** con OpenAPI/Swagger
- **Microservicios async** de alta performance
- **ML/AI endpoints** con validación de datos
- **Equipos que priorizan velocidad** de desarrollo

### Elige Flask para:
- **Aplicaciones web tradicionales** con templates
- **APIs simples** sin complejidad async
- **Proyectos que necesitan control total** del stack
- **Migración desde Flask existente**

## Migración Flask → FastAPI

```python
# Flask existente
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(page=page, per_page=20)
    return jsonify([user.to_dict() for user in users.items])

# Migración a FastAPI
from fastapi import FastAPI, Query
from typing import List

app = FastAPI()

@app.get("/api/users", response_model=List[UserResponse])
async def get_users(page: int = Query(1, ge=1)):
    users = await User.async_paginate(page=page, per_page=20)
    return users
```

## Herramientas complementarias 2025

### FastAPI Stack
```python
# Dependencias recomendadas
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
import httpx

# Async everywhere
async def get_db() -> AsyncSession: ...
async def get_redis() -> Redis: ...
async def get_http_client() -> httpx.AsyncClient: ...
```

### Flask Stack  
```python
# Stack tradicional optimizado
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import requests

# Threading model
db = SQLAlchemy()
cache = Cache()
```

## Conclusión

En 2025, la elección depende de tu contexto:

- **FastAPI**: Si performance, async y documentación automática son prioritarias
- **Flask**: Si necesitas máximo control y tienes un equipo experimentado

Ambos frameworks tienen su lugar y seguirán evolucionando. La clave está en elegir la herramienta correcta para cada proyecto específico.

---
*¿Has migrado de Flask a FastAPI o viceversa? ¡Comparte tu experiencia!*