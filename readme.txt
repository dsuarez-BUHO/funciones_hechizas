Descripción de funciones en: 
    EDA 
    pipeline_functions 

Nota urgente: 
    Importar pipeline_functions es necesario para utilizar cualquier función ya que la función conectar utils se encuentra en este script.
    Sin importar pipeline_functions el kernel se quedará corriendo indefinidamente sin lograr la importación de las funciones deseadas. 

Objetivo de funciones hechizas 
    Generar una serie de funciones clave para facilitar procesos habituales de análisis. 
    Adicionalmente, se busca que las funciones puedan ser empleadas en procesos de automatización. 

Descripción de funciones 
    auto_parse_dates
        utilidad: realiza el cambio de tipo de dato a fecha basado en el nombre de las columnas. Si encuentra la palabra fecha lo cambia. 
        La palabra clave puede ser cambiada ej. "date", "inicio", ...
        
        Sintaxis: 
        utiliza un df, una función de parseo (generalmente: parse_date). 
        El tercer argumento es Keyword="fecha", puede ser sobreescrito. 
    
    

