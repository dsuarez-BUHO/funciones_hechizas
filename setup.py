from setuptools import setup, find_packages
setup(
    name="funciones_hechas",
    version="0.1",
    # Esto ayuda a encontrar paquetes incluso si hay estructuras raras
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "requests",
 #añadir otras si se van ocupando más al hacer funciones que las requieras
    ],
)