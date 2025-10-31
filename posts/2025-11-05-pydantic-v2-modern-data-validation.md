---
layout: post
title: Pydantic V2 - Validación de Datos Moderna en Python
tags: [python, pydantic, validation, data-modeling, fastapi, type-hints]
---

**Pydantic V2** ha revolucionado la validación de datos en Python. Con mejoras de performance del 5-50x y nuevas características, se ha convertido en el estándar para modeling de datos moderno.

## ¿Qué es Pydantic V2?

Una librería de validación de datos que usa **type hints** de Python para definir esquemas, validar datos y serializar/deserializar automáticamente.

### Instalación 2025
```bash
pip install "pydantic>=2.0"

# Con extras para casos específicos
pip install "pydantic[email,url]"  # Validators extras
pip install "pydantic[dotenv]"     # Settings management
```

## Conceptos fundamentales

### 1. BaseModel básico
```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: Optional[int] = Field(None, ge=0, le=120)
    status: StatusEnum = StatusEnum.active
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Configuración del modelo
    model_config = {
        "str_strip_whitespace": True,  # Auto-trim strings
        "validate_default": True,      # Validate default values
        "extra": "forbid"             # No extra fields allowed
    }

# Uso
user_data = {
    "id": 1,
    "name": "  Juan Pérez  ",  # Se auto-trimea
    "email": "juan@example.com",
    "age": 30,
    "tags": ["developer", "python"]
}

user = User(**user_data)
print(user.name)  # "Juan Pérez" (trimmed)
print(user.model_dump())  # Dict with all data
```

### 2. Validators personalizados
```python
from pydantic import BaseModel, field_validator, model_validator
import re

class Product(BaseModel):
    name: str
    price: float
    sku: str
    category: str
    subcategory: Optional[str] = None
    
    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v: str) -> str:
        """SKU debe seguir formato específico"""
        if not re.match(r'^[A-Z]{2}\d{4}$', v):
            raise ValueError('SKU must be format: AA1234')
        return v.upper()
    
    @field_validator('price')
    @classmethod  
    def validate_price(cls, v: float) -> float:
        """Price debe ser positivo y máximo 2 decimales"""
        if v <= 0:
            raise ValueError('Price must be positive')
        return round(v, 2)
    
    @model_validator(mode='after')
    def validate_category_hierarchy(self) -> 'Product':
        """Validación a nivel de modelo completo"""
        electronics_categories = ['smartphone', 'laptop', 'tablet']
        
        if self.category == 'electronics':
            if not self.subcategory:
                raise ValueError('Electronics must have subcategory')
            if self.subcategory not in electronics_categories:
                raise ValueError(f'Invalid subcategory for electronics: {self.subcategory}')
        
        return self

# Uso
try:
    product = Product(
        name="iPhone 15",
        price=999.999,  # Se redondea a 999.00
        sku="ap1234",   # Se convierte a "AP1234"
        category="electronics",
        subcategory="smartphone"
    )
    print(product.model_dump())
except ValidationError as e:
    print(f"Validation error: {e}")
```

### 3. Computed Fields (V2 Feature)
```python
from pydantic import BaseModel, computed_field
from typing import Optional

class Person(BaseModel):
    first_name: str
    last_name: str
    birth_year: Optional[int] = None
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Campo calculado automáticamente"""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def age(self) -> Optional[int]:
        """Edad calculada basada en birth_year"""
        if self.birth_year:
            from datetime import datetime
            return datetime.now().year - self.birth_year
        return None

person = Person(first_name="Ana", last_name="García", birth_year=1990)
print(person.full_name)  # "Ana García"  
print(person.age)        # 35 (calculated)
print(person.model_dump())  # Includes computed fields
```

## Características avanzadas V2

### 1. Field Aliases y Serialization
```python
from pydantic import BaseModel, Field, AliasChoices, AliasPath
from typing import Dict, Any

class APIResponse(BaseModel):
    # Múltiples aliases para flexibilidad
    user_id: int = Field(
        validation_alias=AliasChoices('user_id', 'userId', 'id')
    )
    
    # Nested field access
    email: str = Field(
        validation_alias=AliasPath('user', 'contact', 'email')
    )
    
    # Different serialization name  
    internal_code: str = Field(
        serialization_alias='code'
    )
    
    # Exclude from serialization
    password_hash: str = Field(exclude=True)

# Input data variations all work:
data1 = {"user_id": 123, "user": {"contact": {"email": "test@example.com"}}, 
         "internal_code": "ABC", "password_hash": "secret"}

data2 = {"userId": 123, "user": {"contact": {"email": "test@example.com"}}, 
         "internal_code": "ABC", "password_hash": "secret"}

response = APIResponse(**data1)
print(response.model_dump())  # Uses serialization_alias, excludes password
```

### 2. Custom Serializers
```python
from pydantic import BaseModel, field_serializer
from datetime import datetime
from decimal import Decimal

class Order(BaseModel):
    id: int
    amount: Decimal
    created_at: datetime
    metadata: Dict[str, Any]
    
    @field_serializer('amount')
    def serialize_amount(self, value: Decimal) -> str:
        """Serialize Decimal as string with 2 decimals"""
        return f"{value:.2f}"
    
    @field_serializer('created_at')  
    def serialize_datetime(self, value: datetime) -> str:
        """Custom datetime format"""
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    @field_serializer('metadata', when_used='json')
    def serialize_metadata(self, value: Dict) -> str:
        """Only when serializing to JSON"""
        import json
        return json.dumps(value, sort_keys=True)

order = Order(
    id=1,
    amount=Decimal('99.99'),
    created_at=datetime.now(),
    metadata={"source": "web", "campaign": "summer2025"}
)

print(order.model_dump_json())  # Custom serialized format
```

### 3. Discriminated Unions (Tagged Unions)
```python
from pydantic import BaseModel, Field, Tag
from typing import Union, Literal

class Cat(BaseModel):
    animal_type: Literal['cat'] = 'cat'
    name: str
    lives_remaining: int = Field(ge=0, le=9)

class Dog(BaseModel): 
    animal_type: Literal['dog'] = 'dog'
    name: str
    breed: str
    
class Bird(BaseModel):
    animal_type: Literal['bird'] = 'bird'  
    name: str
    can_fly: bool = True

# Union discriminada por 'animal_type'
Animal = Union[
    Cat,
    Dog, 
    Bird
]

class Pet(BaseModel):
    owner: str
    animal: Animal = Field(discriminator='animal_type')

# Pydantic automáticamente determina el tipo correcto
pets_data = [
    {"owner": "Alice", "animal": {"animal_type": "cat", "name": "Whiskers", "lives_remaining": 7}},
    {"owner": "Bob", "animal": {"animal_type": "dog", "name": "Rex", "breed": "Golden Retriever"}},
    {"owner": "Carol", "animal": {"animal_type": "bird", "name": "Tweety", "can_fly": False}}
]

pets = [Pet(**data) for data in pets_data]
for pet in pets:
    print(f"{pet.owner} owns a {pet.animal.animal_type} named {pet.animal.name}")
```

## Integración con FastAPI

### 1. Request/Response models
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI()

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=18, le=120)
    
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int
    created_at: datetime
    is_active: bool = True
    
    # Config for FastAPI integration
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 1,
                "name": "Juan Pérez",
                "email": "juan@example.com", 
                "age": 30,
                "created_at": "2025-01-01T12:00:00Z",
                "is_active": True
            }
        }
    }

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=50)
    age: Optional[int] = Field(None, ge=18, le=120)

# Endpoints con validación automática
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    # user ya está validado automáticamente
    new_user = await create_user_in_db(user)
    return UserResponse(**new_user)

@app.put("/users/{user_id}", response_model=UserResponse) 
async def update_user(user_id: int, update: UserUpdate):
    # Solo campos no-None se incluyen en update
    update_data = update.model_dump(exclude_unset=True)
    updated_user = await update_user_in_db(user_id, update_data)
    return UserResponse(**updated_user)
```

### 2. Settings management
```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional

class DatabaseConfig(BaseModel):
    host: str = "localhost" 
    port: int = 5432
    name: str
    user: str
    password: str
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class Settings(BaseSettings):
    # Environment variables automaticamente mapeadas
    debug: bool = False
    secret_key: str = Field(..., min_length=32)
    
    # Nested configuration
    database: DatabaseConfig
    
    # Optional external service
    redis_url: Optional[str] = None
    
    # API keys
    stripe_key: Optional[str] = Field(None, alias='STRIPE_SECRET_KEY')
    
    model_config = {
        "env_prefix": "APP_",  # APP_DEBUG, APP_SECRET_KEY, etc.
        "env_nested_delimiter": "__"  # APP_DATABASE__HOST, etc.
    }

# Cargar desde .env automáticamente
settings = Settings()
print(settings.database.url)
```

## Performance y optimization

### 1. Benchmarks Pydantic V2 vs V1
```python
import time
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int

# Performance test  
def benchmark_validation(data_list, iterations=10000):
    start = time.time()
    
    for _ in range(iterations):
        for data in data_list:
            User(**data)
    
    return time.time() - start

test_data = [
    {"id": i, "name": f"User {i}", "email": f"user{i}@example.com", "age": 25+i}
    for i in range(100)
]

duration = benchmark_validation(test_data)
print(f"Validated 1M objects in {duration:.2f}s")

# Pydantic V2: ~2.5s
# Pydantic V1: ~12.0s  
# 5x faster! 🚀
```

### 2. Validation modes para performance
```python
from pydantic import ValidationMode

class HighPerformanceModel(BaseModel):
    # Configuración para máxima velocidad
    model_config = {
        "validate_assignment": False,  # No validate on assignment
        "use_enum_values": True,      # Use enum values directly
        "arbitrary_types_allowed": True,  # Allow any type
        "extra": "ignore"             # Ignore extra fields (faster)
    }

# Validación lazy para lotes grandes
data_batch = [/* large dataset */]

# Validar solo cuando sea necesario
validated_items = []
for item in data_batch:
    try:
        validated = HighPerformanceModel.model_validate(item, strict=False)
        validated_items.append(validated)
    except ValidationError:
        # Log error and continue
        continue
```

## Casos de uso avanzados

### 1. API client con validación
```python
import httpx
from pydantic import BaseModel, ValidationError
from typing import List, Generic, TypeVar

T = TypeVar('T', bound=BaseModel)

class APIClient(Generic[T]):
    def __init__(self, base_url: str, model_class: type[T]):
        self.base_url = base_url
        self.model_class = model_class
        self.client = httpx.AsyncClient()
    
    async def get_all(self, endpoint: str) -> List[T]:
        """Get and validate list of objects"""
        response = await self.client.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        
        data = response.json()
        validated_items = []
        
        for item in data:
            try:
                validated_items.append(self.model_class(**item))
            except ValidationError as e:
                print(f"Validation error for item {item.get('id', '?')}: {e}")
        
        return validated_items

# Uso type-safe
class Product(BaseModel):
    id: int
    name: str  
    price: float

product_client = APIClient[Product]("https://api.shop.com", Product)
products = await product_client.get_all("products")
# products es List[Product] con type checking completo
```

### 2. Configuration management complejo
```python
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os

class ServiceConfig(BaseModel):
    name: str
    url: str
    timeout: int = 30
    retries: int = 3
    
class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: List[str] = ["console", "file"]

class AppSettings(BaseSettings):
    environment: str = "development"
    debug: bool = False
    
    # Multiple service configurations
    services: Dict[str, ServiceConfig] = Field(default_factory=dict)
    
    # Logging configuration  
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Feature flags
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    
    @validator('services', pre=True)
    def parse_services(cls, v):
        """Parse services from environment or config"""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

# Cargar configuración compleja
settings = AppSettings()

# Acceder de forma type-safe
if settings.services.get("payment"):
    payment_service = settings.services["payment"]
    print(f"Payment service: {payment_service.url}")
```

## Mejores prácticas 2025

### ✅ Hacer
- Usar `Field()` para validaciones detalladas
- Implementar `computed_field` para datos derivados
- Usar `discriminator` para unions complejas
- Validar en el edge de tu aplicación
- Aprovechar `model_dump(exclude_unset=True)` para updates

### ❌ Evitar
- Validar datos ya validados repetidamente
- Usar validaciones muy complejas que afecten performance
- Mezclar lógica de negocio con validación
- Ignorar errores de validación silenciosamente

## Conclusión

**Pydantic V2** se ha convertido en la base para:
- **APIs robustas** con FastAPI
- **Data pipelines** confiables
- **Configuration management** type-safe
- **ETL processes** validados

Con mejoras de performance dramáticas y nuevas características como computed fields y discriminated unions, Pydantic V2 es esencial para Python moderno en 2025.

---
*¿Has migrado a Pydantic V2? ¡Comparte tu experiencia con las nuevas características!*