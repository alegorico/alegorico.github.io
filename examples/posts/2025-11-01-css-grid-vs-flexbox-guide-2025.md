---
layout: post
title: CSS Grid vs Flexbox - Guía Definitiva 2021
tags: [css, grid, flexbox, layout, responsive, design]
---

Una de las preguntas más frecuentes en desarrollo frontend: **¿cuándo usar CSS Grid y cuándo Flexbox?** En 2021, ambas tecnologías han madurado y cada una tiene su lugar específico.

## TL;DR - Cuándo usar cada uno

- **Flexbox**: Layouts unidimensionales (filas O columnas)
- **CSS Grid**: Layouts bidimensionales (filas Y columnas)

## Flexbox: El maestro de la dirección única

### Casos de uso perfectos
```css
/* Navegación horizontal */
.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

/* Centrado perfecto */
.modal {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

/* Cards responsivos */
.card-container {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

.card {
  flex: 1 1 300px; /* grow, shrink, basis */
}
```

### Propiedades clave 2021
```css
.flex-container {
  display: flex;
  
  /* Control de dirección */
  flex-direction: row | column;
  
  /* Manejo de espacio */
  justify-content: space-between;
  align-items: center;
  gap: 1rem; /* ¡Ya no más margins! */
  
  /* Responsive sin media queries */
  flex-wrap: wrap;
}
```

## CSS Grid: El arquitecto del layout

### Grid moderno sin media queries
```css
.grid-container {
  display: grid;
  
  /* Auto-responsive */
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  
  /* Areas nombradas */
  grid-template-areas: 
    "header header header"
    "sidebar main aside"
    "footer footer footer";
}

.header { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main { grid-area: main; }
```

### Layouts complejos simplificados
```css
/* Holy Grail Layout en 5 líneas */
.holy-grail {
  display: grid;
  grid-template: auto 1fr auto / 200px 1fr 200px;
  min-height: 100vh;
  gap: 1rem;
}

/* Masonry layout nativo (2021) */
.masonry {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  grid-template-rows: masonry; /* ¡Experimental pero prometedor! */
  gap: 1rem;
}
```

## Combinando ambos: El poder real

```css
/* Grid para layout principal */
.app-layout {
  display: grid;
  grid-template: auto 1fr auto / 250px 1fr;
  min-height: 100vh;
}

/* Flexbox para componentes internos */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
}

.sidebar nav {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
```

## Técnicas avanzadas 2021

### Subgrid (¡Ya disponible!)
```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.card {
  display: grid;
  grid-template-rows: subgrid; /* Alineación perfecta */
  grid-row: span 3;
}
```

### Container Queries + Grid
```css
.component {
  container-type: inline-size;
}

@container (min-width: 500px) {
  .component .grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
```

## Performance y mejores prácticas

### ✅ Hacer
- Usar `gap` en lugar de margins
- Combinar Grid y Flexbox según el contexto
- Aprovechar `auto-fit` y `minmax()` para responsive
- Usar areas nombradas para claridad

### ❌ Evitar
- Grid para todo (overkill en layouts simples)
- Flexbox para layouts complejos 2D
- Media queries cuando `auto-fit` funciona
- Posicionamiento absoluto innecesario

## Herramientas de desarrollo

```css
/* Debug visual */
* {
  outline: 1px solid red;
}

/* CSS Grid Inspector en DevTools */
.grid {
  display: grid;
  /* Firefox/Chrome tienen inspectores visuales */
}
```

## Conclusión

En 2021, la combinación de CSS Grid y Flexbox nos da superpoderes para crear layouts que antes requerían frameworks complejos. La clave está en:

1. **Grid** para la arquitectura general
2. **Flexbox** para componentes y alineación
3. **Container Queries** para responsive moderno
4. **Subgrid** para alineación perfecta

¿Resultado? Menos código, mejor performance, más mantenibilidad.

---
*¿Tienes algún layout complejo que no sabes cómo abordar? ¡Compártelo en los comentarios!*