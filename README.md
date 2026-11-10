# Trabajo Práctico 2 - Procesamiento de Imágenes

Este repositorio contiene la resolución del Trabajo Práctico N°2 de la materia Procesamiento Avanzado de Imágenes

**Alumnos:** Santino Malatini - Gregorio Fernández Perrier - Exequiel Cortesi

## Estructura del repositorio

```text
├── Codigos/                # Scripts de python para resolver los problemas
├── Documentos/             # Enunciado e informe final
├── Imagenes/               
│   ├── Iniciales/          # Imágenes dadas para resolver los problemas
│   └── Resultados/         # Imágenes obtenidas por la resolucion

```


## Guía de Instalación y Ejecución

### 1. Clonar el repositorio
Abre tu terminal y descarga el proyecto localmente:
```bash
git clone https://github.com/SantinoMalatini/procesamiento_de_imagenes_tp2.git
cd procesamiento_de_imagenes_tp2
```

### 2. Crear el entorno virtual
Crea un entorno virtual aislado ejecutando:
```bash
python -m venv venv
```

### 3. Activar el entorno virtual
Activa el entorno virtual ejecutando:
* **Windows:**
  ```bash
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Instalar las dependencias
Con el entorno activado, instala las dependencias ejecutando:
```bash
pip install -r requirements.txt
```

### 5. Ejecución del codigo del problema 1

```bash
python Codigos/problema_1.py
```

### 6. Ejecución del codigo del problema 2

```bash
python Codigos/problema_2.py
```