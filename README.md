# CMS.js - Generador Estático del Lado del Cliente

Este repositorio contiene el código fuente y las herramientas de compilación para **CMS.js**, un motor de generación de sitios estáticos basado en Markdown que se ejecuta completamente en el lado del cliente (Single-Page App).

## Propósito del Paquete

Este paquete se encarga de compilar y optimizar el núcleo de **CMS.js**. El motor toma tus archivos HTML de diseño (layouts), lee publicaciones escritas en Markdown y las renderiza directamente en el navegador del usuario a tiempo de ejecución, eliminando la necesidad de compiladores en el servidor o bases de datos complejas.

## Desarrollo y Compilación

Las herramientas de construcción están basadas en Rollup y Babel. Puedes compilar el motor utilizando los siguientes comandos:

* **Instalar dependencias**:
  ```bash
  npm install
  ```
* **Compilar para producción**:
  ```bash
  npm run build
  ```
  *(Esto genera los archivos listos para usar en `dist/cms.js`, `dist/cms.es.js` y `dist/cms.min.js`)*.
* **Modo de desarrollo**:
  ```bash
  npm run dev
  ```
  *(Levanta un servidor local en el puerto 3000 y compila automáticamente al detectar cambios)*.

---
*🚀 Construido con Rollup, Babel y tecnologías web modernas para la generación del lado del cliente.*
