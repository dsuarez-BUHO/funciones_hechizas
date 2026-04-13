# funciones_hechizas

Librería interna de funciones reutilizables para análisis, EDA, ETL y pipelines de datos.

## Instalación (modo editable)

Clonar el repositorio y luego instalarlo con `pip install -e` para que cualquier cambio en el código se refleje automáticamente sin reinstalar.

```bash
git clone <url-del-repo>
cd funciones_hechizas
pip install -e .
```

Esto registra el paquete como `funciones_hechas` en tu entorno de Python. Desde ese momento se puede importar en cualquier script del mismo entorno.

## Uso

```python
from utils.EDA_functions import *
from utils.ETL_EDA_functions import *
from utils.pipeline_functions import *
from utils.tiempos_campana.tiempos_campana import *
from utils.preproyectos.preproyectos import *
from utils.kpis_wms.kpis_wms_utils import *
```

> **Nota importante:** `pipeline_functions` debe importarse antes de cualquier otro módulo. Contiene la función `conectar_utils` que inicializa las rutas internas del paquete. Sin esta importación el kernel puede quedar corriendo indefinidamente.

## Módulos disponibles

| Módulo | Descripción |
|--------|-------------|
| `utils/EDA_functions.py` | Funciones de análisis exploratorio |
| `utils/ETL_EDA_functions.py` | Funciones de ETL y transformación |
| `utils/pipeline_functions.py` | Funciones de pipeline y conexión de utilidades |
| `utils/tiempos_campana/tiempos_campana.py` | KPIs y análisis de tiempos de campaña |
| `utils/preproyectos/preproyectos.py` | Utilidades para etapa de pre-proyecto |
| `utils/kpis_wms/kpis_wms_utils.py` | KPIs para WMS |

## Agregar dependencias

Si una función nueva requiere una librería adicional, agregarla en `setup.py` bajo `install_requires` y volver a correr `pip install -e .`.
