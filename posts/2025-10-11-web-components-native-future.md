---
layout: post
title: Web Components Nativos - El Futuro Sin Frameworks
tags: [web-components, custom-elements, shadow-dom, html, javascript, future]
---

Los **Web Components** han madurado en 2025 y ofrecen una alternativa poderosa a frameworks complejos. He explorado este paradigma trabajando con proyectos como [framework-less-web-components](https://github.com/alegorico/framework-less-web-components), que demuestra implementaciones puras sin dependencias. Descubre cómo crear componentes reutilizables usando solo APIs nativas del navegador.

## ¿Qué son los Web Components?

Un conjunto de APIs estándar que nos permiten crear elementos HTML personalizados, encapsulados y reutilizables:

- **Custom Elements**: Define nuevos elementos HTML
- **Shadow DOM**: Encapsulación de estilos y markup  
- **HTML Templates**: Plantillas reutilizables
- **ES Modules**: Importación nativa de componentes

## Ventajas sobre frameworks

### 🚀 Performance nativo
- Sin virtual DOM overhead
- Carga solo lo que necesitas
- Cache del navegador optimizado

### 🔒 Encapsulación real  
- Estilos completamente aislados
- No hay conflictos de CSS
- APIs privadas reales

### 📦 Framework agnostic
- Funciona con React, Vue, Angular
- O completamente standalone
- Migración incremental posible

## Mi Experiencia con Framework-less Web Components

En mi fork del proyecto [framework-less-web-components](https://github.com/alegorico/framework-less-web-components), he experimentado con implementaciones puras que demuestran el poder de los Web Components nativos. El proyecto explora patrones sin frameworks para crear componentes robustos y reutilizables.

**Ventajas observadas en proyectos reales:**
- **Zero dependencies**: No runtime overhead de frameworks
- **Interoperabilidad total**: Funcionan en cualquier contexto (React, Vue, vanilla)
- **Longevidad**: APIs estables del browser, no churn de frameworks 
- **Performance nativo**: Renderizado optimizado por el navegador
- **Bundle size**: Componentes minimalistas sin bloat

## Construyendo tu primer componente

### 1. Custom Element básico

```javascript
// alert-box.js
class AlertBox extends HTMLElement {
  constructor() {
    super();
    
    // Crear Shadow DOM
    this.attachShadow({ mode: 'open' });
    
    // Template y estilos
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          padding: 1rem;
          border-radius: 8px;
          margin: 1rem 0;
        }
        
        :host([type="success"]) {
          background: #dcfce7;
          color: #15803d;
          border: 1px solid #bbf7d0;
        }
        
        :host([type="error"]) {
          background: #fee2e2;
          color: #dc2626;
          border: 1px solid #fecaca;
        }
        
        .close-btn {
          float: right;
          background: none;
          border: none;
          font-size: 1.2rem;
          cursor: pointer;
        }
      </style>
      
      <button class="close-btn">&times;</button>
      <slot></slot>
    `;
    
    // Event listeners
    this.shadowRoot.querySelector('.close-btn')
      .addEventListener('click', () => this.remove());
  }
  
  // Lifecycle callbacks
  connectedCallback() {
    console.log('AlertBox conectado al DOM');
  }
  
  disconnectedCallback() {
    console.log('AlertBox removido del DOM');
  }
}

// Registrar el componente
customElements.define('alert-box', AlertBox);
```

### 2. Usando el componente

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./alert-box.js"></script>
</head>
<body>
  <!-- Uso súper simple -->
  <alert-box type="success">
    ¡Operación completada exitosamente!
  </alert-box>
  
  <alert-box type="error">
    Error: No se pudo conectar al servidor
  </alert-box>
</body>
</html>
```

## Componente avanzado: Data Table

```javascript
// data-table.js
class DataTable extends HTMLElement {
  static get observedAttributes() {
    return ['data-url', 'columns'];
  }
  
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.data = [];
  }
  
  connectedCallback() {
    this.render();
    this.loadData();
  }
  
  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'data-url' && newValue !== oldValue) {
      this.loadData();
    }
  }
  
  async loadData() {
    const url = this.getAttribute('data-url');
    if (!url) return;
    
    try {
      const response = await fetch(url);
      this.data = await response.json();
      this.renderTable();
    } catch (error) {
      this.renderError(error.message);
    }
  }
  
  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          font-family: system-ui, -apple-system, sans-serif;
        }
        
        table {
          width: 100%;
          border-collapse: collapse;
          margin: 1rem 0;
        }
        
        th, td {
          padding: 0.75rem;
          text-align: left;
          border-bottom: 1px solid #e5e7eb;
        }
        
        th {
          background: #f9fafb;
          font-weight: 600;
          color: #374151;
        }
        
        tr:hover {
          background: #f9fafb;
        }
        
        .loading {
          text-align: center;
          padding: 2rem;
          color: #6b7280;
        }
        
        .error {
          color: #dc2626;
          padding: 1rem;
          background: #fee2e2;
          border-radius: 4px;
        }
      </style>
      
      <div class="container">
        <div class="loading">Cargando datos...</div>
      </div>
    `;
  }
  
  renderTable() {
    const columns = JSON.parse(this.getAttribute('columns') || '[]');
    const container = this.shadowRoot.querySelector('.container');
    
    container.innerHTML = `
      <table>
        <thead>
          <tr>
            ${columns.map(col => `<th>${col.title}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          ${this.data.map(row => `
            <tr>
              ${columns.map(col => `<td>${row[col.key] || '-'}</td>`).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }
  
  renderError(message) {
    const container = this.shadowRoot.querySelector('.container');
    container.innerHTML = `<div class="error">Error: ${message}</div>`;
  }
}

customElements.define('data-table', DataTable);
```

### Uso del DataTable

```html
<data-table 
  data-url="/api/users"
  columns='[
    {"key": "name", "title": "Nombre"},
    {"key": "email", "title": "Email"},
    {"key": "role", "title": "Rol"}
  ]'>
</data-table>
```

## Patrones avanzados 2025

### 1. Reactive Properties con Proxies

```javascript
class ReactiveComponent extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    
    // State reactivo
    this.state = new Proxy({
      count: 0,
      message: 'Hello World'
    }, {
      set: (target, property, value) => {
        target[property] = value;
        this.render(); // Re-render automático
        return true;
      }
    });
  }
  
  render() {
    this.shadowRoot.innerHTML = `
      <div>
        <p>${this.state.message}</p>
        <p>Count: ${this.state.count}</p>
        <button onclick="this.getRootNode().host.increment()">+</button>
      </div>
    `;
  }
  
  increment() {
    this.state.count++; // Trigger re-render automático
  }
}
```

### 2. Form Components con Validation

```javascript
class ValidatedInput extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.value = '';
    this.errors = [];
  }
  
  static get formAssociated() { return true; }
  
  connectedCallback() {
    this.internals = this.attachInternals();
    this.render();
    this.addEventListeners();
  }
  
  addEventListeners() {
    const input = this.shadowRoot.querySelector('input');
    input.addEventListener('input', (e) => {
      this.value = e.target.value;
      this.validate();
      this.internals.setFormValue(this.value);
    });
  }
  
  validate() {
    this.errors = [];
    
    if (this.hasAttribute('required') && !this.value) {
      this.errors.push('Este campo es requerido');
    }
    
    if (this.hasAttribute('min-length')) {
      const min = parseInt(this.getAttribute('min-length'));
      if (this.value.length < min) {
        this.errors.push(`Mínimo ${min} caracteres`);
      }
    }
    
    const isValid = this.errors.length === 0;
    this.internals.setValidity(
      isValid ? {} : { customError: true }, 
      this.errors.join(', ')
    );
    
    this.renderErrors();
  }
}
```

## Herramientas y ecosystem

### Lit (Lightweight)
```javascript
import { LitElement, html, css } from 'lit';

class SimpleButton extends LitElement {
  static styles = css`
    button {
      background: var(--primary-color, blue);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 4px;
    }
  `;
  
  render() {
    return html`<button @click=${this.handleClick}><slot></slot></button>`;
  }
  
  handleClick() {
    this.dispatchEvent(new CustomEvent('my-click'));
  }
}
```

### Stencil (Compiler)
```typescript
@Component({
  tag: 'my-component',
  styleUrl: 'my-component.css'
})
export class MyComponent {
  @Prop() name: string;
  @State() isVisible = false;
  
  render() {
    return <div>Hello, {this.name}!</div>;
  }
}
```

## Migración estratégica

### 1. Leaf components primero
- Buttons, inputs, modals
- Sin dependencias complejas
- Fácil testing individual

### 2. Micro frontends
- Cada team puede usar su stack
- Comunicación via Custom Events
- Shared design system

### 3. Progressive enhancement
- Empieza con HTML semántico
- Añade interactividad progresivamente
- Funciona sin JavaScript

## Conclusión

Los Web Components en 2025 ofrecen:

- **Performance nativo** sin overhead de frameworks
- **Verdadera encapsulación** con Shadow DOM  
- **Interoperabilidad** total entre tecnologías
- **Longevidad** basada en estándares web

No reemplazan frameworks complejos, pero ofrecen una alternativa poderosa para muchos casos de uso.

---
*¿Has experimentado con Web Components? ¡Comparte tu experiencia!*