## Analisis de relajación termal para diferentes relaciones de masa en un gas AHP

Utilizando (`gas03_analysis.py`) se estudio un gas unidimensional que interactua unicamente por colisiones elasticas para particulas de diferentes masas $m_1$ y $m_2$. Estudiando si al establecer una temperatura inicial $T(x,0)$ en el sistema,  si la temperatura, y otros perfiles hidrodinamicos, cambia como se esperaria en un fluido hidrodinamico. 

Esto con el objetivo de no solo estudiar el cambio en la temperatura, sino que para ver si el cambio de relación de masas afecta cambia estos parámetros también.

Además quisimos probar que nuestro motor efectivamente simula lo que está en el paper Blast in a One Dimensional Cold Gas: From Newtonian Dynamics to Hydrodinamics. Donde, se utiliza un gas unidimensional AHP (Alternating - Hard- PArticle) para simular un fluido hidrodinamico utilizando este gas. 

En este caso, se establecen para el gas condiciones iniciales en la energía y se analizan los perfiles hidrodinamicos de la simulación. Pudimos reproducir estos resultados utilizando nuestro motor. Como en este paper se introduce una condición inicial en la energía $E(x,0)$, se tuvo que editar el código para poder establecer bien la condición inicial con el motor. Esto se hizo con los programas: (`ahp_engine_c.py`) y (`blast_ahp.py`).

Las imagenes de los analisis se encuentran en las respectivas carpetas `\figuras-gas03` y `\figuras-blast` respectivamente. 

